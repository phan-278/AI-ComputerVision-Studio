import cv2
import os
import queue
import threading
import time

from core.tracker import Tracker
from core.reid import ReID
from processors.analytics import AnalyticsTracker
from visualization.visualize import Visualizer
from visualization.dashboard import TopDownDashboard
import numpy as np



# ------------------------------------------------------------------ #
#  Video I/O                                                           #
# ------------------------------------------------------------------ #

def read_video(video_path):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []
    frames = []
    while True:
        try:
            ret, frame = cap.read()
        except Exception as e:
            print(f"Frame read error at frame {len(frames)}: {e}")
            break
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


def get_next_video_name(output_dir="assets/output_videos", prefix="run", ext=".avi"):
    os.makedirs(output_dir, exist_ok=True)
    nums = []
    for f in os.listdir(output_dir):
        if f.startswith(prefix) and f.endswith(ext):
            try:
                nums.append(int(f[len(prefix):-len(ext)]))
            except ValueError:
                pass
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")


def save_video(output_video_frames, output_dir="assets/output_videos"):
    if not output_video_frames:
        print("No frames!")
        return
    if output_video_frames[0] is None:
        print("Invalid frame!")
        return
    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    path = get_next_video_name(output_dir=output_dir)
    out = cv2.VideoWriter(path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print("Saved:", path)


def get_video_fps(source):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 13.0


# ------------------------------------------------------------------ #
#  RTSP Capture — dedicated reader thread                              #
# ------------------------------------------------------------------ #

class RTSPCapture:
    """Reads RTSP stream on a dedicated thread, always keeping the latest frame."""

    def __init__(self, url):
        self.url = url
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {url}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 13.0
        self._q = queue.Queue(maxsize=2)
        self.running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self._cap.read()
            if not ret:
                self.running = False
                break
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)

    def read(self, timeout=5.0):
        try:
            return True, self._q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        self._cap.release()


# ------------------------------------------------------------------ #
#  ReID factory                                                        #
# ------------------------------------------------------------------ #

def build_reid(fps):
    return ReID(
        model_name='osnet_x1_0',
        similarity_threshold=0.75,
        jacket_dist_thresh=50,
        jacket_time_thresh=int(fps * 3),
        long_term_thresh=int(fps * 600),
        makeup_thresh=int(fps * 300),
        stage2_dist_thresh=200,
        gallery_update_interval=5,
        fps=fps,
        device='cpu'
    )


# ------------------------------------------------------------------ #
#  Offline pipeline                                                    #
# ------------------------------------------------------------------ #

def run_offline(config: dict):
    video_path     = config['video_path']
    model_path     = config['model_path']
    stub_path      = config['stub_path']
    reid_stub_path = config['reid_stub_path']
    zones          = config.get('zones', {})
    show_zones     = config.get('show_zones', False)
    output_dir     = config.get('output_dir', 'assets/output_videos')

    print("[Offline] Reading video:", video_path)
    frames = read_video(video_path)
    if not frames:
        return

    fps = get_video_fps(video_path)
    print(f"[Offline] FPS: {fps:.2f} | Total frames: {len(frames)}")

    tracker = Tracker(model_path)
    tracks = tracker.get_object_track(frames, read_from_stub=True, stub_path=stub_path)

    reid_manager = build_reid(fps)
    refined_tracks = reid_manager.load_stub(reid_stub_path)
    if refined_tracks is None:
        print("[Offline] Running ReID...")
        refined_tracks = reid_manager.merge_tracks_offline(frames, tracks)
        reid_manager.save_stub(refined_tracks, reid_stub_path)

    analytics = AnalyticsTracker(fps=fps, mode='offline')
    dashboard = TopDownDashboard(config['map_path'], config['camera_rois'], config['map_rois'])
    
    dashboard_frames = []
    for frame_idx, frame_data in enumerate(refined_tracks.get('person', [])):
        result = analytics.update(frame_idx, frame_data)
        dashboard.update(frame_idx, result)
        dashboard_frames.append(dashboard.render())
    analytics.save_csv(output_dir)

    viz = Visualizer(zones=zones, show_zones=show_zones)
    output_frames = viz.draw_annotations(frames, refined_tracks)
    
    composite = []
    for anno, dash in zip(output_frames, dashboard_frames):
        anno_h, anno_w = anno.shape[:2]
        dash_h, dash_w = dash.shape[:2]
        new_dash_w = int(dash_w * (anno_h / dash_h))
        dash_resized = cv2.resize(dash, (new_dash_w, anno_h))
        comp = np.hstack((anno, dash_resized))
        composite.append(comp)
        
    save_video(composite, output_dir=output_dir)
    print("[Offline] Done!")


# ------------------------------------------------------------------ #
#  Realtime pipeline (RTSP)                                            #
# ------------------------------------------------------------------ #

def run_realtime(config: dict):
    rtsp_url   = config['rtsp_url']
    model_path = config['model_path']
    zones      = config.get('zones', {})
    show_zones = config.get('show_zones', False)
    output_dir = config.get('output_dir', 'assets/output_videos')

    print("[Realtime] Connecting RTSP:", rtsp_url)
    try:
        capture = RTSPCapture(rtsp_url)
    except RuntimeError as e:
        print(e)
        return

    fps = capture.fps
    print(f"[Realtime] Stream FPS: {fps:.2f}")

    tracker      = Tracker(model_path)
    reid_manager = build_reid(fps)
    analytics    = AnalyticsTracker(fps=fps, mode='realtime')
    viz          = Visualizer(zones=zones, show_zones=show_zones)
    dashboard    = TopDownDashboard(config['map_path'], config['camera_rois'], config['map_rois'])

    frame_idx   = 0
    fps_counter = 0
    t_start     = time.time()

    print("[Realtime] Started. Press 'q' to quit.")
    while capture.running:
        ret, frame = capture.read()
        if not ret:
            print("[Realtime] Stream lost.")
            break

        frame_data = tracker.detect_and_track_frame(frame)
        result     = reid_manager.process_frame(frame, frame_data, frame_idx)
        result     = analytics.update(frame_idx, result)

        dashboard.update(frame_idx, result)
        dash = dashboard.render()

        annotated = viz.draw_frame(frame, result)
        
        anno_h, anno_w = annotated.shape[:2]
        dash_h, dash_w = dash.shape[:2]
        new_dash_w = int(dash_w * (anno_h / dash_h))
        dash_resized = cv2.resize(dash, (new_dash_w, anno_h))
        annotated = np.hstack((annotated, dash_resized))

        fps_counter += 1
        elapsed = time.time() - t_start
        if elapsed > 0:
            cv2.putText(annotated, f"FPS: {fps_counter / elapsed:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Studio Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    capture.stop()
    cv2.destroyAllWindows()
    analytics.save_csv(output_dir)
    print("[Realtime] Exited.")

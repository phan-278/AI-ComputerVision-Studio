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


def save_video(output_video_frames, fps=24.0, output_dir="assets/output_videos"):
    if not output_video_frames:
        print("No frames!")
        return
    if output_video_frames[0] is None:
        print("Invalid frame!")
        return
    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    path = get_next_video_name(output_dir=output_dir)
    out = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
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
    """Reads RTSP stream on a dedicated thread with N=3 frame skipping and large buffer."""

    def __init__(self, url, use_tcp=False):
        self.url = url
        self.use_tcp = use_tcp
        self._cap = None
        self._open_capture()
        
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 13.0
        if self.fps <= 0 or self.fps > 100: self.fps = 13.0
        
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 1280
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 720
        
        self._q = queue.Queue(maxsize=100) # Giảm xuống 100 để giảm độ trễ khi bắt đầu presentation
        self.running = True
        self.stream_frame_idx = 0
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open_capture(self):
        if self._cap:
            self._cap.release()
        
        # Ngăn chặn hard crash của FFmpeg khi reconnect liên tục
        os.environ["OPENCV_FFMPEG_THREADS"] = "1"
        
        if self.use_tcp and self.url.startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000"

        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            print(f"[RTSPCapture] Waiting for stream {self.url} to be active...")
            return False
        return True

    def _reader(self):
        empty_retries = 0
        while self.running:
            if self._cap is None or not self._cap.isOpened():
                print("[RTSPCapture] Reconnecting...")
                time.sleep(2)
                self._open_capture()
                continue

            ret, frame = self._cap.read()
            if not ret:
                empty_retries += 1
                if empty_retries > 50:
                    print("[RTSPCapture] Stream timeout, reconnecting...")
                    self._open_capture()
                    empty_retries = 0
                time.sleep(0.01)
                continue
            
            empty_retries = 0
            self.stream_frame_idx += 1
            
            # Frame Skipping (N=3) - Bỏ 2 lấy 1
            if self.stream_frame_idx % 3 != 0:
                continue

            if self._q.full():
                # Block instead of dropping to keep sequence perfect
                # Or drop if we absolutely must, but plan says Queue is large enough
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put((frame, self.stream_frame_idx))

    def read(self, timeout=5.0):
        try:
            return True, self._q.get(timeout=timeout)
        except queue.Empty:
            return False, (None, 0)

    def stop(self):
        self.running = False
        if self._cap:
            self._cap.release()


# ------------------------------------------------------------------ #
#  ReID factory                                                        #
# ------------------------------------------------------------------ #

def build_reid(fps):
    return ReID(
        model_name='osnet_x1_0',
        similarity_threshold=0.75,
        jacket_dist_thresh=50,
        jacket_time_thresh=3.0,
        long_term_thresh=600.0,
        makeup_thresh=300.0,
        stage2_dist_thresh=200,
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

    print("[Offline] Stream Processing (Hướng 1) for:", video_path)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    fps = get_video_fps(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Offline] FPS: {fps:.2f} | Total frames: {total_frames}")

    # Tự động scale các vùng ROI nếu video khác với độ phân giải chuẩn (1280x720)
    scale_x = width / 1280.0
    scale_y = height / 720.0
    if scale_x != 1.0 or scale_y != 1.0:
        print(f"[Offline] Scaling ROIs by x:{scale_x}, y:{scale_y}")
        zones = {k: (v * np.array([scale_x, scale_y])).astype(v.dtype) for k, v in zones.items()}
        config['camera_rois'] = [(roi * np.array([scale_x, scale_y])).astype(roi.dtype) for roi in config['camera_rois']]

    # Load stubs if available
    tracker = Tracker(model_path, fps=fps)
    tracks = None
    if os.path.exists(stub_path):
        import pickle
        with open(stub_path, "rb") as f:
            tracks = pickle.load(f)
            print(f"[Offline] Loaded Tracker stub: {stub_path}")
            
    reid_manager = build_reid(fps)
    refined_tracks = reid_manager.load_stub(reid_stub_path)
    if refined_tracks:
        print(f"[Offline] Loaded ReID stub: {reid_stub_path}")

    analytics = AnalyticsTracker(fps=fps, mode='offline')
    dashboard = TopDownDashboard(config['map_path'], config['camera_rois'], config['map_rois'])
    viz = Visualizer(zones=zones, show_zones=show_zones)

    ret, frame = cap.read()
    if not ret:
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    dash_placeholder = dashboard.render()
    anno_h, anno_w = height, width
    dash_h, dash_w = dash_placeholder.shape[:2]
    new_dash_w = int(dash_w * (anno_h / dash_h))
    comp_width = anno_w + new_dash_w
    comp_height = anno_h

    os.makedirs(output_dir, exist_ok=True)
    out_path = get_next_video_name(output_dir=output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, float(fps), (comp_width, comp_height))

    frame_idx = 0
    build_tracks = []
    build_refined = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx}/{total_frames}...")

        # 1. Tracking
        if tracks is not None:
            frame_data = tracks["person"][frame_idx] if frame_idx < len(tracks["person"]) else {}
        else:
            frame_data = tracker.detect_and_track_frame(frame)
            build_tracks.append(frame_data)
            
        # 2. ReID
        if refined_tracks is not None:
            reid_data = refined_tracks["person"][frame_idx] if frame_idx < len(refined_tracks["person"]) else {}
        else:
            reid_data = reid_manager.process_frame(frame, frame_data, frame_idx)
            build_refined.append(reid_data)
            
        # 3. Analytics
        result = analytics.update(frame_idx, reid_data)
        
        # 4. Dashboard
        dashboard.update(frame_idx, result)
        dash = dashboard.render()
        dash_resized = cv2.resize(dash, (new_dash_w, comp_height))
        
        # 5. Visualization
        annotated = viz.draw_frame(frame, result)
        
        # 6. Composite & Write
        comp = np.hstack((annotated, dash_resized))
        out.write(comp)
        
        frame_idx += 1
        
        # 7. Garbage Collection (Giữ cho RAM và VRAM không bị đầy)
        if frame_idx % 300 == 0:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        
    out.release()
    cap.release()
    
    if tracks is None:
        import pickle
        with open(stub_path, "wb") as f:
            pickle.dump({"person": build_tracks}, f)
            print(f"[Offline] Saved Tracker stub: {stub_path}")
            
    if refined_tracks is None:
        reid_manager.save_stub({"person": build_refined}, reid_stub_path)
        
    analytics.save_csv(output_dir)
    print("[Offline] Done! Saved to:", out_path)


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
        capture = RTSPCapture(rtsp_url, use_tcp=False)
    except RuntimeError as e:
        print(e)
        return

    fps = capture.fps
    print(f"[Realtime] Stream FPS: {fps:.2f}")

    # Tự động scale các vùng ROI nếu stream khác với độ phân giải chuẩn (1280x720)
    scale_x = capture.width / 1280.0
    scale_y = capture.height / 720.0
    if scale_x != 1.0 or scale_y != 1.0:
        print(f"[Realtime] Scaling ROIs by x:{scale_x}, y:{scale_y}")
        zones = {k: (v * np.array([scale_x, scale_y])).astype(v.dtype) for k, v in zones.items()}
        config['camera_rois'] = [(roi * np.array([scale_x, scale_y])).astype(roi.dtype) for roi in config['camera_rois']]

    tracker      = Tracker(model_path, fps=fps)
    reid_manager = build_reid(fps)
    analytics    = AnalyticsTracker(fps=fps, mode='realtime')
    viz          = Visualizer(zones=zones, show_zones=show_zones)
    dashboard    = TopDownDashboard(config['map_path'], config['camera_rois'], config['map_rois'])

    fps_counter = 0
    t_start     = time.time()

    print("[Realtime] Started. Press 'q' to quit.")
    output_frames = []
    
    while capture.running:
        ret, (frame, stream_frame_idx) = capture.read()
        if not ret:
            print("[Realtime] Stream lost or timeout.")
            continue

        frame_data = tracker.detect_and_track_frame(frame)
        result     = reid_manager.process_frame(frame, frame_data, stream_frame_idx)
        result     = analytics.update(stream_frame_idx, result)

        dashboard.update(stream_frame_idx, result)
        dash = dashboard.render()

        annotated_cam = viz.draw_frame(frame, result)
        
        # --- Premium Layout: Dashboard as master, Camera feed smaller ---
        target_h = 800
        dh, dw = dash.shape[:2]
        dash_resized = cv2.resize(dash, (int(dw * target_h / dh), target_h))
        
        # Camera is smaller (60% of dashboard height) to "see the studio" better
        cam_h = int(target_h * 0.6)
        ch, cw = annotated_cam.shape[:2]
        cam_resized = cv2.resize(annotated_cam, (int(cw * cam_h / ch), cam_h))
        
        # Pad camera feed with a nice dark border to match dashboard height
        pad_h = target_h - cam_h
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        cam_padded = cv2.copyMakeBorder(cam_resized, pad_top, pad_bot, 10, 10, cv2.BORDER_CONSTANT, value=(30, 30, 35))
        
        # Horizontal stack: [Camera Padded] | [Dashboard]
        composite = np.hstack((cam_padded, dash_resized))

        fps_counter += 1
        elapsed = time.time() - t_start
        if elapsed > 0:
            cv2.putText(composite, f"FPS: {fps_counter / elapsed:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Studio Tracking - Realtime", composite)
        output_frames.append(composite)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.stop()
    cv2.destroyAllWindows()
    
    if output_frames:
        print(f"[Realtime] Finished. Exporting results to {output_dir}...")
        # Since we skip 2 out of 3 frames, effective FPS is fps / 3.0
        effective_fps = fps / 3.0
        save_video(output_frames, fps=effective_fps, output_dir=output_dir) # Saves as runX.avi by default
        
    analytics.save_csv(output_dir) # Saves as analyticsX.csv
    print("[Realtime] All files exported successfully. Exited.")

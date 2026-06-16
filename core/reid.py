import pickle
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from collections import deque
from torchvision import transforms
import torchreid
from utils.bbox_utils import get_foot_position, measure_distance, get_cosine_similarity

# Tọa độ vùng cầu thang (cửa duy nhất của studio)
STAIRCASE_POLY = np.array([[1077, 225], [1245, 487], [950, 511], [809, 252]], dtype=np.int32)

# Tọa độ ROI đi vào khu vực makeup
MAKEUP_POLY    = np.array([[208, 641], [174, 719], [1269, 716], [1275, 649]], dtype=np.int32)


def is_in_staircase(pos):
    return cv2.pointPolygonTest(STAIRCASE_POLY, (float(pos[0]), float(pos[1])), False) >= 0


def is_in_makeup_zone(pos):
    return cv2.pointPolygonTest(MAKEUP_POLY, (float(pos[0]), float(pos[1])), False) >= 0


class ReID:
    def __init__(self,
                 model_name='osnet_x1_0',
                 similarity_threshold=0.75,
                 jacket_dist_thresh=50,       # px
                 jacket_time_thresh=3.0,      # seconds
                 long_term_thresh=600.0,      # seconds
                 makeup_thresh=300.0,         # seconds
                 stage2_dist_thresh=200,      # px
                 fps=13.0,
                 device='cpu'):

        self.similarity_threshold = similarity_threshold
        self.jacket_dist_thresh = jacket_dist_thresh
        self.jacket_time_thresh_frames = int(jacket_time_thresh * fps)
        self.long_term_thresh_frames = int(long_term_thresh * fps)
        self.makeup_thresh_frames = int(makeup_thresh * fps)
        self.stage2_dist_thresh = stage2_dist_thresh
        self.fps = fps
        self.max_feature_samples = 10  # Lưu tối đa 10 feature để tính trung bình

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        ).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # State dùng chung cho cả offline lẫn realtime
        self.gallery = {}       # {actual_id: {features: list, mean_feature: ndarray, feature_locked: bool, ...}}
        self.id_map = {}        # {bytetrack_id: actual_id}
        self.next_id = 1

    def reset(self):
        """Reset toàn bộ state — dùng trước khi bắt đầu offline processing mới."""
        self.gallery = {}
        self.id_map = {}
        self.next_id = 1

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_feature(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        # Tối ưu Resize: dùng cv2 thay vì PIL/torchvision
        crop = cv2.resize(crop, (128, 256))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(input_tensor)
            feature = F.normalize(feature, p=2, dim=1)
        return feature.cpu().numpy().flatten()

    # ------------------------------------------------------------------ #
    #  Matching                                                            #
    # ------------------------------------------------------------------ #

    def _match_with_gallery(self, current_feat, current_pos, current_frame):
        current_in_staircase = is_in_staircase(current_pos)
        current_in_makeup = is_in_makeup_zone(current_pos)
        best_id, best_score = None, -1

        for track_id, data in self.gallery.items():
            time_gap = current_frame - data['last_frame']
            out_zone = data.get('out_zone')
            lost_status = data.get('status', 'hidden')

            # Timeout theo zone: makeup = 5 phút, còn lại = 10 phút
            effective_thresh = self.makeup_thresh_frames if out_zone == 'makeup' else self.long_term_thresh_frames
            if time_gap > effective_thresh:
                continue

            dist = measure_distance(current_pos, data['last_pos'])

            # Phase 1: Spatial-Temporal Gating — không cần feature
            if time_gap < self.jacket_time_thresh_frames and dist < self.jacket_dist_thresh:
                return track_id, 1.0

            # Phase 2: Feature matching
            if current_feat is None:
                continue

            if data.get('feature_locked') and data.get('mean_feature') is not None:
                # Dùng vector trung bình để so khớp chính xác
                max_sim = get_cosine_similarity(current_feat, data['mean_feature'])
            else:
                if not data['features']:
                    continue
                sims = [get_cosine_similarity(current_feat, f) for f in data['features']]
                max_sim = max(sims)

            # Stage 1: Strict
            if max_sim >= 0.85:
                if max_sim > best_score:
                    best_score, best_id = max_sim, track_id

            # Stage 2: Relaxed + ngữ cảnh
            elif max_sim >= 0.70:
                if current_in_staircase and lost_status == 'out' and out_zone == 'staircase':
                    # Quay lại từ vệ sinh / ra ngoài
                    score = max_sim * 0.95
                    if score > best_score:
                        best_score, best_id = score, track_id
                elif current_in_makeup and lost_status == 'out' and out_zone == 'makeup':
                    # Quay lại từ khu makeup
                    score = max_sim * 0.95
                    if score > best_score:
                        best_score, best_id = score, track_id
                elif not current_in_staircase and not current_in_makeup and lost_status == 'hidden':
                    # Occlusion ngắn giữa phòng
                    if dist < self.stage2_dist_thresh:
                        score = max_sim * 0.90
                        if score > best_score:
                            best_score, best_id = score, track_id

        return best_id, best_score

    # ------------------------------------------------------------------ #
    #  Gallery management                                                  #
    # ------------------------------------------------------------------ #

    def _update_gallery(self, actual_id, feature, foot_pos, frame_idx):
        if actual_id not in self.gallery:
            self.gallery[actual_id] = {
                'features': [],
                'mean_feature': None,
                'feature_locked': False,
                'last_pos': foot_pos,
                'last_frame': frame_idx,
                'entry_frame': frame_idx,
                'status': 'active',
                'out_zone': None
            }
        
        # Chỉ cập nhật feature nếu chưa locked
        data = self.gallery[actual_id]
        if feature is not None and not data['feature_locked']:
            data['features'].append(feature)
            # Nếu đủ số lượng mẫu, tính trung bình và khoá lại
            if len(data['features']) >= self.max_feature_samples:
                mean_feat = np.mean(data['features'], axis=0)
                mean_feat = mean_feat / np.linalg.norm(mean_feat) # Normalize lại
                data['mean_feature'] = mean_feat
                data['feature_locked'] = True
                
        data['last_pos'] = foot_pos
        data['last_frame'] = frame_idx
        data['status'] = 'active'
        data['out_zone'] = None

    def mark_lost_tracks(self, active_ids, current_frame):
        for actual_id, data in self.gallery.items():
            if actual_id not in active_ids and data['status'] == 'active':
                if is_in_staircase(data['last_pos']):
                    data['status'] = 'out'
                    data['out_zone'] = 'staircase'
                elif is_in_makeup_zone(data['last_pos']):
                    data['status'] = 'out'
                    data['out_zone'] = 'makeup'
                else:
                    data['status'] = 'hidden'
                    data['out_zone'] = None

    # ------------------------------------------------------------------ #
    #  Stub: lưu / tải trạng thái ReID để debug nhanh                     #
    # ------------------------------------------------------------------ #

    def save_stub(self, refined_tracks, stub_path):
        # Chuyển deque → list để pickle được
        gallery_serial = {
            pid: {**data, 'features': list(data['features'])}
            for pid, data in self.gallery.items()
        }
        payload = {
            'refined_tracks': refined_tracks,
            'gallery': gallery_serial,
            'next_id': self.next_id,
            'fps': self.fps,
        }
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[ReID Stub] Saved → {stub_path}")

    def load_stub(self, stub_path):
        if not os.path.exists(stub_path):
            return None
        with open(stub_path, 'rb') as f:
            payload = pickle.load(f)
        # Restore deque từ list
        for data in payload['gallery'].values():
            data['features'] = deque(data['features'], maxlen=self.max_feature_samples)
        self.gallery = payload['gallery']
        self.next_id = payload['next_id']
        print(f"[ReID Stub] Loaded ← {stub_path}")
        return payload['refined_tracks']

    # ------------------------------------------------------------------ #
    #  Online: xử lý từng frame (realtime)                                #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame, frame_data, frame_idx):
        """
        Xử lý 1 frame cho realtime streaming.
        Tối ưu CPU: Rule 10-frame. Dừng trích xuất sau khi đủ 10 mẫu.
        """
        active_actual_ids = set()
        result = {}

        for old_id, track_info in frame_data.items():
            bbox = track_info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            foot_pos = get_foot_position(bbox)
            actual_id = self.id_map.get(old_id)
            is_new = actual_id is None
            feature = None

            if is_new:
                # ID mới từ ByteTrack → cần feature để match gallery
                feature = self.extract_feature(frame, bbox)
                matched_id, _ = self._match_with_gallery(feature, foot_pos, frame_idx)
                actual_id = matched_id if matched_id is not None else self.next_id
                if matched_id is None:
                    self.next_id += 1
                self.id_map[old_id] = actual_id
            else:
                # ID đã biết → chỉ extract feature nếu chưa bị lock (chưa đủ 10 frames)
                if actual_id in self.gallery and not self.gallery[actual_id].get('feature_locked', False):
                    feature = self.extract_feature(frame, bbox)

            self._update_gallery(actual_id, feature, foot_pos, frame_idx)
            active_actual_ids.add(actual_id)

            entry_frame = self.gallery[actual_id]['entry_frame']
            stay_seconds = (frame_idx - entry_frame) / self.fps

            result[actual_id] = {
                'bbox': bbox,
                'stay_duration': stay_seconds
            }

        self.mark_lost_tracks(active_actual_ids, frame_idx)
        return result

    # ------------------------------------------------------------------ #
    #  Offline: xử lý toàn bộ video sau khi đã có tracks                 #
    # ------------------------------------------------------------------ #

    def merge_tracks_offline(self, frames, tracks):
        self.reset()
        person_tracks = tracks.get("person", [])
        new_person_tracks = []
        total = len(person_tracks)

        for frame_idx, frame_data in enumerate(person_tracks):
            if frame_idx % 300 == 0:
                print(f"  ReID: frame {frame_idx}/{total}")

            frame = frames[frame_idx]
            active_actual_ids = set()
            result = {}

            for old_id, track_info in frame_data.items():
                bbox = track_info['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                foot_pos = get_foot_position(bbox)
                feature = self.extract_feature(frame, bbox)  # Offline: extract mọi frame

                actual_id = self.id_map.get(old_id)
                if actual_id is None:
                    matched_id, _ = self._match_with_gallery(feature, foot_pos, frame_idx)
                    actual_id = matched_id if matched_id is not None else self.next_id
                    if matched_id is None:
                        self.next_id += 1
                    self.id_map[old_id] = actual_id

                self._update_gallery(actual_id, feature, foot_pos, frame_idx)
                active_actual_ids.add(actual_id)

                entry_frame = self.gallery[actual_id]['entry_frame']
                stay_seconds = (frame_idx - entry_frame) / self.fps

                result[actual_id] = {
                    'bbox': bbox,
                    'stay_duration': stay_seconds
                }

            self.mark_lost_tracks(active_actual_ids, frame_idx)
            new_person_tracks.append(result)

        return {"person": new_person_tracks}

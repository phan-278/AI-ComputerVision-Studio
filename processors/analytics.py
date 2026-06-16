import csv
import os
from datetime import datetime


class AnalyticsTracker:
    """Tracks stay duration per person and exports a summary CSV at session end."""

    def __init__(self, fps: float, mode: str = 'offline'):
        self.fps = fps
        self.mode = mode
        self._first_seen_time: dict = {}   # realtime: {pid: datetime}
        self._first_seen_frame: dict = {}  # offline:  {pid: int}
        self._last_seen_frame: dict = {}   # offline:  {pid: int}
        self._last_stay: dict = {}         # {pid: float seconds}

    def update(self, frame_idx: int, person_dict: dict) -> dict:
        """
        Call every frame with the current person_dict from ReID.
        Uses stream_frame_idx for precise time tracking.
        """
        result = {}
        for pid, info in person_dict.items():
            if pid not in self._first_seen_frame:
                self._first_seen_frame[pid] = frame_idx
            self._last_seen_frame[pid] = frame_idx
            
            # ReID already computed stay_duration based on (current_frame - entry_frame) / fps
            stay = info.get('stay_duration', 0.0)
            
            result[pid] = info
            self._last_stay[pid] = stay

        return result

    def _frame_to_hms(self, frame: int) -> str:
        secs = frame / self.fps
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        return f'{h:02d}:{m:02d}:{s:02d}'


    def get_next_analytics_name(self, output_dir="assets/output_videos", prefix="analytics", ext=".csv"):
        os.makedirs(output_dir, exist_ok=True)

        nums = []
        for f in os.listdir(output_dir):
            if f.startswith(prefix) and f.endswith(ext):
                try:
                    num_part = f[len(prefix):-len(ext)]  # lấy phần số
                    nums.append(int(num_part))
                except ValueError:
                    pass

        next_num = max(nums) + 1 if nums else 1
        return os.path.join(output_dir, f"{prefix}{next_num}{ext}")


    def save_csv(self, output_dir: str = 'assets/output_videos') -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = self.get_next_analytics_name(output_dir=output_dir)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['person_id', 'first_seen', 'last_seen', 'stay_seconds', 'stay_hms'])
            
            for pid in sorted(self._last_stay):
                secs = self._last_stay[pid]
                h = int(secs // 3600)
                m = int((secs % 3600) // 60)
                s = int(secs % 60)
                hms = f'{h:02d}:{m:02d}:{s:02d}'
                
                first = self._frame_to_hms(self._first_seen_frame.get(pid, 0))
                last  = self._frame_to_hms(self._last_seen_frame.get(pid, 0))
                writer.writerow([f'P{pid}', first, last, f'{secs:.1f}', hms])
        print(f'[Analytics] Saved → {path}')
        return path

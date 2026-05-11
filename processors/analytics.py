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
        - offline : passes stay_duration through unchanged (already computed by reid),
                    records first/last seen frame for CSV timestamps.
        - realtime: replaces stay_duration with wall-clock elapsed time.
        Returns an updated person_dict ready for the Visualizer.
        """
        now = datetime.now() if self.mode == 'realtime' else None

        result = {}
        for pid, info in person_dict.items():
            if self.mode == 'realtime':
                if pid not in self._first_seen_time:
                    self._first_seen_time[pid] = now
                stay = (now - self._first_seen_time[pid]).total_seconds()
                result[pid] = {**info, 'stay_duration': stay}
            else:
                if pid not in self._first_seen_frame:
                    self._first_seen_frame[pid] = frame_idx
                self._last_seen_frame[pid] = frame_idx
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

    def save_csv(self, output_dir: str = 'assets/output_videos') -> str:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(output_dir, f'analytics_{ts}.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if self.mode == 'offline':
                writer.writerow(['person_id', 'first_seen', 'last_seen', 'stay_seconds', 'stay_hms'])
            else:
                writer.writerow(['person_id', 'stay_seconds', 'stay_hms'])
            for pid in sorted(self._last_stay):
                secs = self._last_stay[pid]
                h = int(secs // 3600)
                m = int((secs % 3600) // 60)
                s = int(secs % 60)
                hms = f'{h:02d}:{m:02d}:{s:02d}'
                if self.mode == 'offline':
                    first = self._frame_to_hms(self._first_seen_frame.get(pid, 0))
                    last  = self._frame_to_hms(self._last_seen_frame.get(pid, 0))
                    writer.writerow([f'P{pid}', first, last, f'{secs:.1f}', hms])
                else:
                    writer.writerow([f'P{pid}', f'{secs:.1f}', hms])
        print(f'[Analytics] Saved → {path}')
        return path

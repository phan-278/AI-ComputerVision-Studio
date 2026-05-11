import cv2
import numpy as np


class Visualizer:
    """Vẽ annotation lên frame: bounding box label, thời gian, zone overlay."""

    def __init__(self,
                 label_color=(255, 0, 255),
                 text_color=(0, 0, 0),
                 zones=None,
                 show_zones=False):
        self.label_color = label_color
        self.text_color = text_color
        # zones = {'staircase': np.array(...), 'makeup': np.array(...)}
        self.zones = {k: np.array(v, dtype=np.int32) for k, v in (zones or {}).items()}
        self.show_zones = show_zones
        self._zone_colors = {
            'staircase': (0, 165, 255),   # cam
            'makeup': (0, 255, 0),         # xanh lá
        }

    # ------------------------------------------------------------------ #
    #  Vẽ label P{id}: HH:MM:SS lên 1 bounding box                       #
    # ------------------------------------------------------------------ #

    def draw_track_id(self, frame, bbox, track_id, stay_duration=None):
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) // 2

        if stay_duration is not None:
            h = int(stay_duration // 3600)
            m = int((stay_duration % 3600) // 60)
            s = int(stay_duration % 60)
            label = f"P{track_id}: {h:02d}:{m:02d}:{s:02d}"
            rect_w = 145
        else:
            label = f"P{track_id}"
            rect_w = 50

        rect_x1 = x_center - rect_w // 2
        rect_y1 = y1 - 30
        rect_x2 = x_center + rect_w // 2
        rect_y2 = y1 - 8

        rect_y1 = max(rect_y1, 0)
        rect_y2 = max(rect_y2, rect_y1 + 1)

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), self.label_color, cv2.FILLED)
        cv2.putText(frame, label, (rect_x1 + 4, rect_y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text_color, 1)
        return frame

    # ------------------------------------------------------------------ #
    #  Vẽ polygon zone lên frame để debug                                  #
    # ------------------------------------------------------------------ #

    def draw_zones(self, frame):
        for name, poly in self.zones.items():
            color = self._zone_colors.get(name, (255, 255, 255))
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
            centroid = poly.mean(axis=0).astype(int)
            cv2.putText(frame, name, tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    # ------------------------------------------------------------------ #
    #  Realtime: vẽ annotation cho 1 frame                                #
    # ------------------------------------------------------------------ #

    def draw_frame(self, frame, person_dict):
        frame = frame.copy()
        if self.show_zones:
            frame = self.draw_zones(frame)
        for track_id, person in person_dict.items():
            frame = self.draw_track_id(frame, person['bbox'], track_id,
                                       person.get('stay_duration'))
        return frame

    # ------------------------------------------------------------------ #
    #  Offline: vẽ annotation cho toàn bộ video                           #
    # ------------------------------------------------------------------ #

    def draw_annotations(self, frames, tracks):
        person_tracks = tracks.get("person", [])
        output = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            if self.show_zones:
                frame = self.draw_zones(frame)
            if frame_num < len(person_tracks):
                for track_id, person in person_tracks[frame_num].items():
                    frame = self.draw_track_id(frame, person['bbox'], track_id,
                                               person.get('stay_duration'))
            output.append(frame)
        return output

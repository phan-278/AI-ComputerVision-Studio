import cv2
import numpy as np
from core.transform import PerspectiveTransformer
from processors.heatmap_gen import HeatmapGenerator

class TopDownDashboard:
    def __init__(self, map_path, camera_rois, map_rois):
        self.map_img = cv2.imread(map_path)
        if self.map_img is None:
            # Fallback if map not found
            self.map_img = np.zeros((945, 755, 3), dtype=np.uint8)
        else:
            self.map_img = cv2.resize(self.map_img, (755, 945))
        
        self.transformer = PerspectiveTransformer(camera_rois, map_rois)
        self.heatmap = HeatmapGenerator()
        
        self.current_positions = {} # {id: (x, y)}
        self.stay_durations = {} # {id: duration}

    def update(self, frame_idx, person_dict):
        self.current_positions.clear()
        self.stay_durations.clear()
        
        valid_positions = []
        for track_id, track_info in person_dict.items():
            bbox = track_info['bbox']
            pos = self.transformer.transform_point(bbox)
            if pos is not None:
                self.current_positions[track_id] = pos
                track_info['position_transformed'] = pos
                
                valid_positions.append(pos)
                if 'stay_duration' in track_info:
                    self.stay_durations[track_id] = track_info['stay_duration']
                    
        self.heatmap.update(valid_positions)

    def _format_time(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def render(self):
        # The base map is 755x945
        map_base = self.map_img.copy()
        
        # Overlay heatmap
        map_base = self.heatmap.get_overlay(map_base, alpha=0.6)
        
        # Draw Title on map
        title = "STUDIO"
        overlay = map_base.copy()
        cv2.rectangle(overlay, (0, 0), (755, 60), (0, 0, 0), -1)
        map_base = cv2.addWeighted(overlay, 0.7, map_base, 0.3, 0)
        cv2.putText(map_base, title, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # Draw each person on map
        for track_id, pos in self.current_positions.items():
            x, y = pos
            
            try:
                numeric_id = int(str(track_id).replace('P', ''))
            except ValueError:
                numeric_id = hash(track_id) % 10000

            np.random.seed(numeric_id)
            color = tuple(int(c) for c in np.random.randint(50, 255, 3))
            
            # Glow effect
            overlay_glow = map_base.copy()
            cv2.circle(overlay_glow, (x, y), 20, color, -1)
            map_base = cv2.addWeighted(overlay_glow, 0.3, map_base, 0.7, 0)
            
            # Dot
            cv2.circle(map_base, (x, y), 8, color, -1)
            cv2.circle(map_base, (x, y), 10, (255, 255, 255), 2)
            
            # Text label
            duration = self.stay_durations.get(track_id, 0)
            time_str = self._format_time(duration)
            label = f"P{numeric_id} | {time_str}"
            
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 15
            text_y = y + 5
            
            overlay_text = map_base.copy()
            cv2.rectangle(overlay_text, (text_x - 4, text_y - text_h - 4), (text_x + text_w + 4, text_y + baseline + 2), (0, 0, 0), -1)
            map_base = cv2.addWeighted(overlay_text, 0.7, map_base, 0.3, 0)
            
            cv2.rectangle(map_base, (text_x - 4, text_y - text_h - 4), (text_x + text_w + 4, text_y + baseline + 2), color, 1)
            cv2.putText(map_base, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Create full canvas with extra space at bottom
        panel_height = 180
        canvas = np.zeros((945 + panel_height, 755, 3), dtype=np.uint8)
        
        # Place map on top
        canvas[:945, :] = map_base
        
        # Draw bottom panel background
        cv2.rectangle(canvas, (0, 945), (755, 945 + panel_height), (20, 20, 25), -1)
        cv2.line(canvas, (0, 945), (755, 945), (100, 100, 100), 2)
        cv2.putText(canvas, "ACTIVE PEOPLE", (20, 980), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw IDs in bottom panel
        if self.current_positions:
            col_width = 180
            row_height = 40
            start_x = 20
            start_y = 1025
            
            for i, track_id in enumerate(sorted(self.current_positions.keys())):
                try:
                    numeric_id = int(str(track_id).replace('P', ''))
                except ValueError:
                    numeric_id = hash(track_id) % 10000
                np.random.seed(numeric_id)
                color = tuple(int(c) for c in np.random.randint(50, 255, 3))
                
                duration = self.stay_durations.get(track_id, 0)
                time_str = self._format_time(duration)
                
                # Arrange: từ trên xuống dưới (3 IDs mỗi cột), rồi từ trái sang phải
                row = i % 3
                col = i // 3
                
                cx = start_x + col * col_width
                cy = start_y + row * row_height
                
                # Dot
                cv2.circle(canvas, (cx, cy - 5), 6, color, -1)
                cv2.circle(canvas, (cx, cy - 5), 7, (255, 255, 255), 1)
                
                # ID and Time
                cv2.putText(canvas, f"P{numeric_id}", (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(canvas, time_str, (cx + 80, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)

        return canvas

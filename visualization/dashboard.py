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

    def render(self):
        base = self.map_img.copy()
        
        # Overlay heatmap
        base = self.heatmap.get_overlay(base, alpha=0.6)
        
        # Draw each person
        for track_id, pos in self.current_positions.items():
            x, y = pos
            
            # Make sure id is int to use for random seed
            try:
                numeric_id = int(str(track_id).replace('P', ''))
            except ValueError:
                numeric_id = hash(track_id) % 10000

            np.random.seed(numeric_id)
            color = tuple(int(c) for c in np.random.randint(50, 255, 3))
            
            # Chấm tròn r=10, viền trắng
            cv2.circle(base, (x, y), 10, color, -1)
            cv2.circle(base, (x, y), 10, (255, 255, 255), 2)
            
            # Nhãn P{id}
            label = f"P{numeric_id}"
            if track_id in self.stay_durations:
                duration = self.stay_durations[track_id]
                label += f": {duration:.1f}s"
                
            cv2.putText(base, label, (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(base, label, (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return base

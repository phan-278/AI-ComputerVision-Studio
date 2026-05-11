import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self, camera_rois, map_rois):
        """
        camera_rois: List of 2 numpy arrays [lower_roi, upper_roi] from camera view.
        map_rois: List of 2 numpy arrays [lower_roi, upper_roi] from map (top-down) view.
        """
        self.camera_rois = camera_rois
        self.H = []
        for i in range(2):
            h_matrix, _ = cv2.findHomography(camera_rois[i], map_rois[i])
            self.H.append(h_matrix)

    def transform_point(self, bbox):
        """
        Takes a bounding box [x1, y1, x2, y2].
        Returns (x, y) on the map or None if out of bounds.
        """
        x1, y1, x2, y2 = bbox
        foot_pos = ((x1 + x2) / 2.0, float(y2))
        head_pos = ((x1 + x2) / 2.0, float(y1))

        # Check upper ROI (index 1) with foot_pos
        if cv2.pointPolygonTest(self.camera_rois[1], foot_pos, False) >= 0:
            pt = np.array([[foot_pos]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.H[1])
            return int(transformed[0][0][0]), int(transformed[0][0][1])
        
        # Check lower ROI (index 0) with head_pos
        if cv2.pointPolygonTest(self.camera_rois[0], head_pos, False) >= 0:
            pt = np.array([[head_pos]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.H[0])
            return int(transformed[0][0][0]), int(transformed[0][0][1])

        return None

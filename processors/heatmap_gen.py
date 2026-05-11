import cv2
import numpy as np

class HeatmapGenerator:
    def __init__(self, shape=(945, 755)):
        # shape is (height, width) -> (945, 755)
        self.shape = shape
        self.accumulator = np.zeros(self.shape, dtype=np.float32)

    def update(self, positions):
        for pos in positions:
            x, y = pos
            if 0 <= x < self.shape[1] and 0 <= y < self.shape[0]:
                self.accumulator[y, x] += 1.0

    def get_overlay(self, base_img, alpha=0.6):
        if np.max(self.accumulator) == 0:
            return base_img
        
        # Gaussian Blur
        blurred = cv2.GaussianBlur(self.accumulator, (0, 0), 30)
        
        # Normalize to 0-255
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply COLORMAP_JET
        heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        
        # Alpha blend only where there is activity
        mask = norm > 0
        
        overlay = base_img.copy()
        overlay[mask] = (base_img[mask] * (1 - alpha) + heatmap_color[mask] * alpha).astype(np.uint8)
        
        return overlay

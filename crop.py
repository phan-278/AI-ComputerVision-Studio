import cv2
import numpy as np
import matplotlib.pyplot as plt


CAMERA_ROIS = [
    np.array([[1216,472],[1274,716],[233,716],[344,490]], dtype=np.int32),
    np.array([[521,146],[342,491],[1220,475],[902,146]], dtype=np.int32),
]


def draw_rois(image, rois, color=(0,255,0), thickness=3):

    output = image.copy()

    for i, roi in enumerate(rois):

        pts = roi.reshape((-1,1,2))

        cv2.polylines(
            output,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness
        )

        cx = int(np.mean(roi[:,0]))
        cy = int(np.mean(roi[:,1]))

        cv2.putText(
            output,
            f"ROI {i}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    return output


# đọc ảnh
image = cv2.imread("assets/input_videos/last_frame.jpg")

# vẽ ROI
result = draw_rois(image, CAMERA_ROIS)

# OpenCV dùng BGR -> chuyển sang RGB
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# hiển thị
plt.figure(figsize=(12,8))
plt.imshow(result_rgb)
plt.axis("off")
plt.show()
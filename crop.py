import cv2
import numpy as np
import matplotlib.pyplot as plt


CAMERA_ROIS  = [
    np.array([[1277, 473], [1272, 714], [175, 716], [272, 508]], dtype=np.float32),
    np.array([[506, 131], [270, 527], [1278, 482], [910, 145]], dtype=np.float32),
]


def draw_rois(image, rois, color=(0,255,0), thickness=3):

    output = image.copy()

    for i, roi in enumerate(rois):

        # IMPORTANT
        pts = roi.astype(np.int32).reshape((-1,1,2))

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

if image is None:
    raise ValueError("Không đọc được ảnh")

# vẽ ROI
result = draw_rois(image, CAMERA_ROIS)

# BGR -> RGB
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# hiển thị
plt.figure(figsize=(12,8))
plt.imshow(result_rgb)
plt.axis("off")
plt.show()
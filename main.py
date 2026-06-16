import numpy as np
from utils.video_utils import run_offline, run_realtime

# ======================= CONFIG ==========================
MODE = "realtime"           # "offline" or "realtime"

VIDEO_PATH     = "assets/input_videos/sample_input.mp4"
MODEL_PATH     = "models/best.pt"
STUB_PATH      = "assets/stubs/track_stub.pkl"
REID_STUB_PATH = "assets/stubs/reid_stub.pkl"
RTSP_URL       = "rtsp://localhost:8554/studio"
OUTPUT_DIR     = "assets/output_videos"
SHOW_ZONES     = True

STAIRCASE_POLY = np.array([[1077, 225], [1245, 487], [950, 511], [809, 252]], dtype=np.int32)
MAKEUP_POLY    = np.array([[208, 641], [174, 719], [1269, 716], [1275, 649]], dtype=np.int32)

MAP_PATH     = "assets/input_videos/map_studio.jpg"
CAMERA_ROIS  = [
    np.array([[1277, 473], [1272, 714], [175, 716], [272, 508]], dtype=np.float32),  # lower
    np.array([[506, 131], [270, 527], [1278, 482], [910, 145]],  dtype=np.float32),  # upper
]
MAP_ROIS     = [
    np.array([[660,663],[695,837],[94,833],[123,660]],    dtype=np.float32),  # lower
    np.array([[170,373],[121,660],[663,663],[610,372]],   dtype=np.float32),  # upper
]
# =========================================================

CONFIG = {
    'video_path':     VIDEO_PATH,
    'model_path':     MODEL_PATH,
    'stub_path':      STUB_PATH,
    'reid_stub_path': REID_STUB_PATH,
    'rtsp_url':       RTSP_URL,
    'output_dir':     OUTPUT_DIR,
    'show_zones':     SHOW_ZONES,
    'zones': {
        'staircase': STAIRCASE_POLY,
        'makeup':    MAKEUP_POLY,
    },
    'map_path': MAP_PATH,
    'camera_rois': CAMERA_ROIS,
    'map_rois': MAP_ROIS,
}

if __name__ == "__main__":
    if MODE == "offline":
        run_offline(CONFIG)
    elif MODE == "realtime":
        run_realtime(CONFIG)
    else:
        print(f"MODE không hợp lệ: {MODE}. Dùng 'offline' hoặc 'realtime'.")

# 📖 1. Description
**AI Computer Vision Studio** is an intelligent video processing and tracking system designed for studio environments. It tracks people within specific zones, maps their positions onto a static 2D floor plan using homography, and generates heatmaps to analyze foot traffic, all while optimizing processing for CPU-only execution.

# 🏷️ 2. Badges/Tags
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

# 🚀 3. Intro
Welcome to the AI Computer Vision Studio project! This project was built as part of the "Introduction to Computer Vision" course. It aims to provide real-time and offline video analysis for physical spaces, allowing studio managers to track customer movements, monitor specific zones (like the makeup area or staircase), and visualize data through heatmaps and 2D map projections.

# 📂 4. Project Structure
```text
AI-ComputerVision-Studio/
├── 📁 assets/                 # Input videos, stubs, output videos, maps
│   ├── 📁 input_videos/       # Input videos for processing
│   ├── 📁 output_videos/      # Output videos with tracking overlays
│   └── 📁 stubs/              # Stubs for data processing(saving for debug)
├── 📁 core/                   # Core logic for tracking and transformation
│   ├── 📄 reid.py             # Handles Person Re-Identification (ReID) features
│   ├── 📄 tracker.py          # Object tracking integrating YOLO and ByteTrack
│   └── 📄 transform.py        # Homography perspective transformation (3D to 2D map)
├── 📁 ffmpeg/                 # FFmpeg binaries and scripts for video processing
│   ├── 📄 ffmpeg.exe          
│   ├── 📄 ffplay.exe          
│   └── 📄 ffprobe.exe         
├── 📁 mediamtx/               # RTSP server configurations for live streaming
│   ├── 📄 mediamtx.exe        
│   └── 📄 mediamtx.yml          
├── 📁 models/                 # Pre-trained AI models
│   └── 📄 best.pt             # Trained YOLO weights
├── 📁 processors/             # Data processing modules
│   ├── 📄 analytics.py        # Analyzes and logs data (e.g., zone interactions)
│   └── 📄 heatmap_gen.py      # Generates spatial heatmaps based on foot traffic
├── 📁 utils/                  # Helper utility functions
│   ├── 📄 bbox_utils.py       # Math functions (distances, foot positions, cosine sim)
│   └── 📄 video_utils.py      # Video stream reading, threading, and RTSP handling
├── 📁 visualization/          # Drawing and rendering modules
│   ├── 📄 dashboard.py        # Renders the top-down 2D dashboard/map
│   └── 📄 visualize.py        # Overlays bounding boxes, zones, and heatmaps on frames
├── 📄 crop.py                 # Utility script for cropping ROIs from videos/images
├── 📄 main.py                 # Main execution script (Offline / Realtime modes)
├── 📄 requirements.txt        # List of Python dependencies
└── 📄 README.md               # Project documentation
```

# 💻 5. Technologies
- **Programming Language:** 🐍 Python
- **Computer Vision:** 👁️ OpenCV
- **Object Detection & Tracking:** 🎯 YOLO, ByteTrack
- **Re-Identification (ReID):** 👤 Custom ReID for tracking individuals across frames
- **Math & Matrix Operations:** 🧮 NumPy

# ✨ 6. Features
- **Real-time & Offline Processing:** Support for both pre-recorded video files and live RTSP streams.
- **Zone Monitoring:** Define polygonal regions of interest (ROIs) like 'staircase' or 'makeup' areas to track interactions.
- **Homography Mapping:** Convert perspective video frames into a top-down static 2D map.
- **Heatmap Generation:** Visualize the most frequented areas in the studio over time.
- **Optimized for CPU:** Uses frame skipping and ByteTrack + ReID combination to achieve smooth tracking without a dedicated GPU.

# ⌨️ 7. Keyboard Shortcuts
- `q`: Quit the video stream or stop processing.

# ⚙️ 8. The Process
Building this system involved significant optimization challenges. Since the video processing runs entirely on a CPU, heavy trackers like StrongSORT were not feasible. To solve this, I integrated **ByteTrack with ReID** and implemented **frame skipping** to drastically reduce the computational load. 
Another major challenge was handling ID switches (swap ID) when customers wore similar colored clothing, which complicated the feature matching process.

# 🧠 9. What I Learned
Through this project, I gained hands-on experience with:
- 📊 Collecting data and training custom AI models.
- 🖼️ Implementing advanced image processing techniques.
- 🔗 Integrating ReID (Re-Identification) for robust multi-object tracking.
- 🗺️ Understanding and applying Homography to transform video perspectives onto a static 2D map.
- 🌡️ Generating heatmaps to visualize spatial data.
- 📡 Handling and processing live video streams.

# 🚀 10. How to Improve
In the future, I plan to expand the system by adding **Behavior Detection**. Specifically, I want to implement models that can detect if customers are bringing prohibited items like outside food or drinks into the studio.

# 🛠️ 11. How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/phan-278/AI-ComputerVision-Studio
   cd AI-ComputerVision-Studio
   ```
2. **Download Large Assets & Tools:**
   Large files and executables are not pushed to this repository. Please download the `models/best.pt`, sample videos, `ffmpeg.exe`, and `mediamtx.exe` from this **[Link Google Drive / Release]** (update link here) and extract them into their respective folders (`models/`, `assets/`, `ffmpeg/`, `mediamtx/`).
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   Configure the `main.py` file to set `MODE = "offline"` or `"realtime"`, then execute:
   split bash into 3:

   1.bash
   cd mediamtx
   .\mediamtx.exe 
   ```

   2.bash
   .\ffmpeg\bin\ffmpeg.exe -re -stream_loop -1 -i assets\input_videos\sample_input.mp4 -c copy -f rtsp rtsp://localhost:8554/studio
   ```

   3.bash
   python main.py
   ```

# 🎥 12. Video Demo
📹 1-minute setup and execution guide: [demo.mp4](assets/input_videos/demo.mp4)

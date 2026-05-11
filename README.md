
tracks["persons"][frame_num][track_id] = {
    "bbox": [x1, y1, x2, y2],
    "position": (x, y),                # Pixel (foot position) trên ảnh gốc
    "position_transformed": (x_2d, y_2d), # Tọa độ trên mặt bằng Studio (Bird-view)
    
    # --- Thông tin bổ sung quan trọng cho Studio ---
    "entry_timestamp": float,          # Thời điểm bắt đầu xuất hiện (giây thứ mấy)
    "stay_duration": float,            # Thời gian đã ở lại tính đến frame này (giây)
    "features": np.array,              # Feature vector từ OSNet (dùng để ReID)
    "is_active": bool                  # Người này còn trong khung hình hay không
}
Tôi đang làm đô án thời gian khách ở lại(từ lúc xuất hiện đến lúc rời) studio và heatmap(top-down view của studio) realtime
thông tin studio
-diện tích khoảng 80m2, 1 camera duy nhất,camera cố định gốc quay từ trên xuống bao quát cả studio, máy sử dụng cpu không có gpu,
-số lượng người thuê studio , tức là người ở phòng đó không quá 10 người
tracking: bytetrack, osnet(reid)
input: 1 video(stream từ camera)
output: 1 video đã xử lí(id được đánh trên đầu người và thời gian họ ở studio) + dashboard phải video (hiển thị top-down viestudio_analytics/
├── assets/                 # Chứa video test, ảnh sơ đồ studio (bird-view)
│   ├── input_videos/
│   ├── output_videos/
│   └── stubs/              # chứa thông tin tracker,reid để chạy nhanh hơn dễ debug
├── model/                  # Chứa file .pt của YOLOv11s và OSNet
│   └── best.pt             # yolo11s
├── core/                   # "Trái tim" của hệ thống
│   ├── tracker.py          # Cấu hình ByteTrack
│   ├── reid.py             # Logic trích xuất feature & Gallery matching
│   └── transform.py        # Xử lý Homography (Perspective to Bird-view(top-down view))
├── processors/
│   ├── heatmap_gen.py      # Thuật toán vẽ Heatmap (Gaussian Blur)
│   └── analytics.py        # Tính toán thời gian ở lại (Stay duration)
├── visualization
│   └── visualize.py        # thực hiện việc vẽ
├── utils/                  # Các hàm bổ trợ (vẽ box, xử lý video frame)
│   ├── bbox_utils.py 
│   └── video_utils.py      # Đọc video, lưu video
│
├── main.py                 # Script chính chạy 

pipeline từ Camera/Videow của studio heatmap lun top-down view này + P{id} : time tức là id của người đó và thời gian họ ở studio)
code reid phải xử lí được trường hợp khách cởi áo khoát ra và khách đi vệ sinh(mất id đi vệ sinh khoảng 5-10p), giữ id của khách xuyên suốt video
Hãy bắt đầu bằng việc thực hiện chức năng giữ id xuyên suốt video trước chưa cần thực hiện chức năng dashboard(heatmap, top-down view các thứ) chỉ cần output ra 1 video đã  xử lí là được
Tôi đang chạy video offline để debug trước chưa cần vào realtime ngay
Trước khi bắt đầu hãy hỏi tôi 10 câu hỏi để cải thiện code và hiểu rõ hơn đồ án này

# ============================================


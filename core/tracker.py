import supervision as sv
from ultralytics import YOLO
import pickle
import os
import sys
sys.path.append("../")

class Tracker():
    def __init__(self, model_path, fps=13.0):
        self.model = YOLO(model_path)
        # ByteTrack tuning for N=3 frame skipping (effective ~4.3 FPS)
        # Higher lost_track_buffer (e.g. 130 frames = 10 seconds of source video)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25, 
            lost_track_buffer=130, 
            minimum_matching_threshold=0.8
        )
        self._person_cls_id = None  # Cache class id của 'person'
        
    def get_object_detection(self,frames):
        batch_size = 20
        detections = []

        for i in range(0,len(frames),batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size])
            detections += detection_batch

        return detections
    
    def get_object_track(self,frames, read_from_stub = False, stub_path = None):

        # read from stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,"rb") as f:
                tracks = pickle.load(f)
            
            return tracks
        
        detections = self.get_object_detection(frames)

        tracks = {"person":[]}

        for frame_num, detection in enumerate(detections):
            
            cls_name = detection.names
            cls_name_inv = {v:k for k,v in cls_name.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            detection_with_track = self.tracker.update_with_detections(detection_supervision)

            tracks["person"].append({})

            for frame_detection in detection_with_track:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                # Only track persons
                if cls_id == cls_name_inv.get('person', 0):
                    tracks["person"][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def detect_and_track_frame(self, frame):
        """Detect + track 1 frame duy nhất — dùng cho realtime streaming."""
        detection = self.model.predict(frame, imgsz=640, verbose=False)[0]

        # Cache person class id lần đầu
        if self._person_cls_id is None:
            cls_name_inv = {v: k for k, v in detection.names.items()}
            self._person_cls_id = cls_name_inv.get('person', 0)

        detection_supervision = sv.Detections.from_ultralytics(detection)
        detection_with_track = self.tracker.update_with_detections(detection_supervision)

        result = {}
        for frame_detection in detection_with_track:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]
            if cls_id == self._person_cls_id:
                result[track_id] = {'bbox': bbox}
        return result
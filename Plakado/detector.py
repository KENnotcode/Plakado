# plakado/detector.py

from ultralytics import YOLO
import cv2

class PlateDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thres=0.25):
        self.model = YOLO(model_path)
        self.conf = conf_thres

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                detections.append({"bbox": (x1, y1, x2, y2), "crop": crop})
        return detections

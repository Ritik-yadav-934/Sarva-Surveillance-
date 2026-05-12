from ultralytics import YOLO
import supervision as sv

class ProductDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model(frame)[0]

        detections = sv.Detections.from_ultralytics(results)

        return detections
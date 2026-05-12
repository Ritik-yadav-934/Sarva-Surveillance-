from ultralytics import YOLO

class YOLODetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model(frame)

        detections = []

        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls
                })

        return detections
import cv2
from yolo_model import YOLODetector

print("Loading model...")

model = YOLODetector("models/best.pt")

print("Opening video...")

cap = cv2.VideoCapture("data/videos/test.mp4")

if not cap.isOpened():
    print("❌ Error: Cannot open video")
    exit()

print("✅ Video opened successfully")

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ No frame received / video ended")
        break

    print("✅ Frame received")

    detections = model.detect(frame)

    print(f"Detections: {len(detections)}")

    for det in detections:

        x1, y1, x2, y2 = map(int, det["bbox"])

        conf = det["confidence"]
        cls = det["class_id"]

        label = f"Class:{cls} Conf:{conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    resized_frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
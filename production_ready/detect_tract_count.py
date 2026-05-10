# This is what factories actually want — count rolls passing a line:

from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture("factory_video.mp4")

counted_ids = set()         # store unique roll IDs
total_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        verbose=False
    )

    # Count each unique ID once
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        for track_id in ids:
            if track_id not in counted_ids:
                counted_ids.add(track_id)
                total_count += 1

    # Draw + show count
    annotated = results[0].plot()
    cv2.putText(annotated, f"Total Rolls: {total_count}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    cv2.imshow("Rolled Object Counter", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Total rolls detected: {total_count}")
# This is the real deployment — uses your laptop camera or factory CCTV stream:

from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

# 0 = laptop webcam
# Or use RTSP URL for CCTV: "rtsp://username:password@192.168.1.10:554/stream"
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.4, verbose=False)
    
    # Draw boxes on frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Rolled Object Detection", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
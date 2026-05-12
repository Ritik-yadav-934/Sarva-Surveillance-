import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================
model = YOLO("models/best.pt")

# =========================
# INITIALIZE TRACKER
# =========================
tracker = sv.ByteTrack()

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture("data/videos/test.mp4")

# =========================
# CREATE WINDOW
# =========================
cv2.namedWindow("Polygon Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Polygon Counter", 1280, 720)

# =========================
# DEFINE POLYGON ZONE
# =========================
polygon = np.array([
    [400, 200],
    [1000, 200],
    [1000, 700],
    [400, 700]
])

zone = sv.PolygonZone(
    polygon=polygon,
    frame_resolution_wh=(1280, 720)
)

# =========================
# COUNTER VARIABLES
# =========================
counted_ids = set()

total_count = 0

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (1280, 720))

    # =========================
    # YOLO DETECTION
    # =========================
    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    # =========================
    # TRACKING
    # =========================
    detections = tracker.update_with_detections(detections)

    # =========================
    # ZONE CHECK
    # =========================
    in_zone = zone.trigger(detections)

    # =========================
    # DRAW POLYGON
    # =========================
    cv2.polylines(
        frame,
        [polygon],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3
    )

    # =========================
    # PROCESS DETECTIONS
    # =========================
    for i, (bbox, track_id, class_id) in enumerate(zip(
        detections.xyxy,
        detections.tracker_id,
        detections.class_id
    )):

        x1, y1, x2, y2 = map(int, bbox)

        class_name = model.names[class_id]

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        # Label
        label = f"{class_name} | ID:{track_id}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        # =========================
        # COUNT LOGIC
        # =========================
        if in_zone[i]:

            if track_id not in counted_ids:

                counted_ids.add(track_id)

                total_count += 1

                print(f"Counted ID: {track_id}")

    # =========================
    # DISPLAY COUNT
    # =========================
    cv2.putText(
        frame,
        f"TOTAL COUNT: {total_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0,0,255),
        3
    )

    # =========================
    # SHOW FRAME
    # =========================
    cv2.imshow("Polygon Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
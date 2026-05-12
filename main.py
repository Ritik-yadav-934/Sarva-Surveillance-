# import cv2
# import supervision as sv
# from ultralytics import YOLO

# # =========================
# # LOAD YOLO MODEL
# # =========================
# print("Loading YOLO model...")

# model = YOLO("models/best.pt")

# print("Model loaded successfully")


# # =========================
# # INITIALIZE TRACKER
# # =========================
# tracker = sv.ByteTrack()

# print("Tracker initialized")


# # =========================
# # OPEN VIDEO
# # =========================
# video_path = "data/videos/test.mp4"

# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("❌ Error: Cannot open video")
#     exit()

# print("✅ Video opened successfully")


# # =========================
# # CREATE RESIZABLE WINDOW
# # =========================
# cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Tracking", 1280, 720)


# # =========================
# # MAIN LOOP
# # =========================
# while True:

#     ret, frame = cap.read()

#     if not ret:
#         print("Video ended")
#         break

#     # =========================
#     # YOLO INFERENCE
#     # =========================
#     results = model(frame)[0]

#     # =========================
#     # CONVERT DETECTIONS
#     # =========================
#     detections = sv.Detections.from_ultralytics(results)

#     # =========================
#     # UPDATE TRACKER
#     # =========================
#     detections = tracker.update_with_detections(detections)

#     # =========================
#     # DRAW TRACKED OBJECTS
#     # =========================
#     for bbox, track_id, class_id in zip(
#         detections.xyxy,
#         detections.tracker_id,
#         detections.class_id
#     ):

#         x1, y1, x2, y2 = map(int, bbox)

#         # CLASS NAMES
#         class_name = model.names[class_id]

#         # DRAW BOX
#         cv2.rectangle(
#             frame,
#             (x1, y1),
#             (x2, y2),
#             (0, 255, 0),
#             2
#         )

#         # LABEL
#         label = f"{class_name} | ID: {track_id}"

#         cv2.putText(
#             frame,
#             label,
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 0),
#             2
#         )

#     # =========================
#     # SHOW FRAME
#     # =========================
#     cv2.imshow("Tracking", frame)

#     # PRESS Q TO EXIT
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


# # =========================
# # RELEASE RESOURCES
# # =========================
# cap.release()
# cv2.destroyAllWindows()

import cv2

from detection.detector import ProductDetector
from tracking.tracker import ProductTracker
from counting.counter import ZoneCounter


# =========================
# INITIALIZE MODULES
# =========================
detector = ProductDetector("models/best.pt")

tracker = ProductTracker()

counter = ZoneCounter()


# =========================
# VIDEO SOURCE
# =========================
cap = cv2.VideoCapture("data/videos/test.mp4")

cv2.namedWindow("Industrial AI System", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Industrial AI System", 1280, 720)


# =========================
# MAIN PIPELINE LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # =========================
    # DETECTION
    # =========================
    detections = detector.detect(frame)

    # =========================
    # TRACKING
    # =========================
    tracked_detections = tracker.update(detections)

    # =========================
    # COUNTING
    # =========================
    total_count = counter.process(
        tracked_detections
    )

    # =========================
    # DRAW OBJECTS
    # =========================
    for bbox, track_id, class_id in zip(
        tracked_detections.xyxy,
        tracked_detections.tracker_id,
        tracked_detections.class_id
    ):

        x1, y1, x2, y2 = map(int, bbox)

        class_name = detector.model.names[class_id]

        # Bounding box
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
    # DRAW POLYGON
    # =========================
    cv2.polylines(
        frame,
        [counter.polygon],
        True,
        (0,0,255),
        3
    )

    # =========================
    # SHOW COUNT
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
    cv2.imshow("Industrial AI System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

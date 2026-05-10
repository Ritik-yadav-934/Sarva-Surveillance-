from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="factory_video.mp4",
    save=True,
    conf=0.4
)
print("✅ Output saved in: runs/detect/predict/")
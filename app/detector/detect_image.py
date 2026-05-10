from ultralytics import YOLO

# Load your model
model = YOLO("best.pt")

# Detect on an image
results = model.predict(
    source="my_factory_photo.jpg",
    save=True,         # saves annotated image
    conf=0.4,          # confidence threshold
    show=True          # opens preview window
)

# Print what was detected
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"Found: {label} ({conf:.2%} confidence)")
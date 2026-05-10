# Build a Web App (Share with Anyone)
# Use Streamlit — make a web app in 30 lines:


import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("🏭 Rolled Object Detector")
st.write("Upload an image to detect rolled objects")

model = YOLO("best.pt")

uploaded = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original", width=400)

    if st.button("🔍 Detect"):
        results = model.predict(image, conf=0.4)
        annotated = results[0].plot()[:, :, ::-1]  # BGR → RGB
        st.image(annotated, caption="Detected", width=400)
        
        st.success(f"Found {len(results[0].boxes)} rolled objects")
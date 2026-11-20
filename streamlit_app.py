import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="PPE Object Detector", page_icon="🦺", layout="wide")
st.title("🦺 Construction PPE Object Detector")
st.write("Upload an image to detect objects using a fine-tuned YOLO model.")

MODEL_PATH = Path("models/best.pt")

@st.cache_resource
def load_model(path):
    return YOLO(str(path))

if not MODEL_PATH.exists():
    st.error("Model file not found. Add your trained model at `models/best.pt`.")
    st.info("Train using the notebook, then copy `best.pt` into the repository's models folder.")
    st.stop()

model = load_model(MODEL_PATH)
confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    results = model.predict(image, conf=confidence, verbose=False)
    result = results[0]
    annotated = result.plot()
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Detection results")
        st.image(annotated, channels="BGR", use_container_width=True)
    with right:
        st.subheader("Detected objects")
        rows = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                rows.append({
                    "Object": model.names[cls_id],
                    "Confidence": round(float(box.conf.item()), 4),
                    "Confidence (%)": f"{float(box.conf.item())*100:.1f}%"
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.metric("Objects detected", len(rows))
        else:
            st.info("No objects detected at this confidence threshold.")
else:
    st.caption("Choose an image above to start detecting.")

from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Page setup
st.set_page_config(page_title="PCB Defect Detector", layout="centered")
st.title("🔍 PCB Defect Detector")

# Load model weights relative to your app.py directory
@st.cache_resource
def load_model():
  return YOLO("best.pt")


model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload a PCB Image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)

  # Layout columns for side-by-side view
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

  with col2:
    st.subheader("Detection Result")
    with st.spinner("Running detection..."):
      # Run inference
      results = model(image)
      # Render results as NumPy array
      res_plotted = results[0].plot()
      st.image(res_plotted, use_container_width=True)
      
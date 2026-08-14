import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="PCB Defect Inspector",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Automated PCB Defect Detection System")
st.markdown("Perform real-time visual inspection on Printed Circuit Boards using YOLOv8.")

# 2. Sidebar Settings
st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.10, 1.00, 0.45, 0.05)

input_type = st.sidebar.radio("Select Input Method", ["Upload Image", "Use Camera"])

# 3. Cache and Load Model
@st.cache_resource
def load_model():
    try:
        return YOLO("best.onnx")
    except Exception:
        try:
            return YOLO("best.pt")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None

model = load_model()

# 4. Input Handling
uploaded_file = None
if input_type == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.sidebar.camera_input("Take a photo of the PCB")

if uploaded_file is not None and model is not None:
    # Safely convert image format
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_container_width=True)

    # 5. Model Inference
    with st.spinner("Analyzing PCB surface..."):
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=1024
        )[0]

    # Plot output bounding boxes
    res_plotted = results.plot()
    res_image = Image.fromarray(res_plotted[:, :, ::-1])

    with col2:
        st.subheader("🎯 Inspection Output")
        st.image(res_image, use_container_width=True)

    # 6. Detailed Defect Analytics Report
    st.markdown("---")
    st.subheader("📊 Inspection Report")

    boxes = results.boxes
    if len(boxes) > 0:
        names = model.names
        detected_classes = [names[int(cls)] for cls in boxes.cls]
        confidences = [round(float(conf), 3) for conf in boxes.conf]

        report_df = pd.DataFrame({
            "Defect Category": detected_classes,
            "Confidence Score": confidences
        })

        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.metric("Total Defects Detected", len(boxes))
            summary_counts = report_df["Defect Category"].value_counts().reset_index()
            summary_counts.columns = ["Defect Type", "Count"]
            st.dataframe(summary_counts, hide_index=True)

            # --- CSV Export Feature ---
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Report as CSV",
                data=csv,
                file_name="pcb_defect_report.csv",
                mime="text/csv",
            )

        with col_right:
            st.dataframe(report_df, use_container_width=True)
    else:
        st.success("✅ Clean PCB! No manufacturing defects detected.")
else:
    st.info("👈 Provide a PCB image from the sidebar to inspect defects.")
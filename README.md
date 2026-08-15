# 🔬 Automated PCB Defect Detection System

An AI-powered computer vision web application built using **YOLOv8** and **Streamlit** for real-time visual inspection and defect detection on Printed Circuit Boards (PCBs).

🔗 **Live Web Demo:** [AI PCB Defect Detector App](https://aipcbdefectdetector-qz36pecwebhs9flg8s6l5m.streamlit.app)

---

## 🌟 Key Features

* **Dual Input Modes:** Support for uploading high-resolution PCB images or capturing real-time snapshots via webcam.
* **Real-time Anomaly Detection:** Detects and localizes manufacturing defects (solder bridges, missing holes, open circuits, short circuits, etc.) using custom YOLOv8 models.
* **Interactive Adjustments:** Real-time sliders for adjusting **Confidence Threshold** and **IoU Threshold** dynamically.
* **QA Data Logging:** Automatically generates summary metrics and an interactive tabular log of detected anomalies.
* **CSV Report Export:** Enables Quality Assurance (QA) inspectors to download structured inspection logs with a single click.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Computer Vision & Deep Learning:** Ultralytics YOLOv8, OpenCV, Pillow, ONNX Runtime
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Deployment:** Streamlit Community Cloud

---

## 🚀 Local Installation & Setup

To run this project locally on your machine, follow these steps:

 **Clone the repository:**
   ```bash
   git clone https://github.com/madhurisettipalli7/AI_PCB_DEFECT_DETECTOR.git
   cd AI_PCB_DEFECT_DETECTOR
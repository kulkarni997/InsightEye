import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from datetime import datetime

# 1. Professional Page Setup
st.set_page_config(page_title="Sentinel-Vision AI", layout="wide")
st.markdown("<style>main {background-color: #0e1117; color: #ffffff;}</style>", unsafe_allow_html=True)

# 2. Sidebar Controls
st.sidebar.title("🛡️ System Control")
conf_threshold = st.sidebar.slider("AI Sensitivity (Confidence)", 0.0, 1.0, 0.45)
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("Engine: YOLOv11 Optimized")
st.sidebar.info("Mode: Real-time Edge Inference")

# 3. Initialize Model
@st.cache_resource # This keeps the app fast
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# 4. Main Interface
st.title("Sentinel-Vision: Neural Object Intelligence")
col1, col2 = st.columns([2, 1])

img_file_buffer = st.camera_input("Initialize Scan")

if img_file_buffer is not None:
    # Process Image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # AI Inference
    results = model(cv2_img, conf=conf_threshold)

    with col1:
        st.subheader("Neural Overlay Feed")
        annotated_frame = results[0].plot()
        st.image(annotated_frame, channels="BGR")

    with col2:
        st.subheader("Telemetry Data")
        labels = [model.names[int(box.cls[0])] for box in results[0].boxes]
        
        if labels:
            df = pd.DataFrame(labels, columns=["Object"]).value_counts().reset_index()
            df.columns = ["Object Class", "Count"]
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Add a timestamped log
            st.write(f"**Last Scan:** {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.warning("No threats or objects detected.")
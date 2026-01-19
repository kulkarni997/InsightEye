import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from datetime import datetime
from fpdf import FPDF # New Import
import tempfile

# --- NEW FUNCTION: PDF GENERATION ---
def create_pdf(df, img_bgr):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Header
    pdf.cell(190, 10, "Sentinel-Vision Scan Report", ln=True, align='C')
    pdf.set_font("Arial", "", 10)
    pdf.cell(190, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    # Add Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(95, 10, "Object Class", border=1)
    pdf.cell(95, 10, "Count", border=1, ln=True)
    
    pdf.set_font("Arial", "", 12)
    for index, row in df.iterrows():
        pdf.cell(95, 10, str(row['Object Class']), border=1)
        pdf.cell(95, 10, str(row['Count']), border=1, ln=True)
    
    # Save image to a temporary file to embed in PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
        # Convert BGR to RGB for the report
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(tmpfile.name, img_bgr)
        pdf.ln(10)
        pdf.image(tmpfile.name, x=10, w=180)
        
    return pdf.output()

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
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# 4. Main Interface
st.title("Sentinel-Vision: Neural Object Intelligence")
col1, col2 = st.columns([2, 1])

with st.container():
    img_file_buffer = st.camera_input("Initialize Scan")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    results = model(cv2_img, conf=conf_threshold)

    with col1:
        st.subheader("Neural Overlay Feed")
        annotated_frame = results[0].plot()
        st.image(annotated_frame, channels="BGR", use_container_width=True)

    with col2:
        st.subheader("Telemetry Data")
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            labels = [model.names[int(box.cls[0])] for box in boxes]
            df = pd.DataFrame(labels, columns=["Object Class"]).value_counts().reset_index()
            df.columns = ["Object Class", "Count"]
            
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # --- ACTION BUTTONS ---
            c1, c2 = st.columns(2)
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            c1.download_button(
                label="📥 CSV Report",
                data=csv,
                file_name=f"scan_{datetime.now().strftime('%H%M%S')}.csv",
                mime='text/csv',
            )
            
            # PDF Download
            pdf_data = create_pdf(df, annotated_frame)
            c2.download_button(
                label="📄 PDF Report",
                data=pdf_data,
                file_name=f"sentinel_report_{datetime.now().strftime('%H%M%S')}.pdf",
                mime='application/pdf'
            )
            
            st.write(f"**Last Scan:** {datetime.now().strftime('%H:%M:%S')}")
            st.write(f"**Total Objects:** {len(labels)}")
        else:
            st.warning("No threats or objects detected so far.")
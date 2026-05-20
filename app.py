import streamlit as st
import numpy as np
import cv2
from streamlit_cropper import st_cropper
from PIL import Image

# Page configuration
st.set_page_config(page_title="Porous Surface Roughness Analyzer", layout="centered")

st.title("📊 Porous Material Surface Roughness Analyzer")
st.write("Maps and quantifies localized surface roughness variations across complex multi-layered porous structures.")

# Sidebar Controls
st.sidebar.header("Roughness Calculation Settings")

window_size = st.sidebar.slider(
    "Local Texture Window Size", 
    min_value=5, 
    max_value=51, 
    value=15, 
    step=2,
    help="The pixel neighborhood size used to evaluate texture roughness. Smaller windows catch micro-roughness."
)

intensity_scaling = st.sidebar.slider(
    "Z-Axis Max Calibration Value (µm)", 
    min_value=1.0, 
    max_value=500.0, 
    value=100.0,
    help="Maps the maximum pixel intensity change to a physical depth scale in micrometers."
)

uploaded_file = st.file_uploader("Upload material image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    
    st.subheader("1. Define Target Evaluation Region")
    cropped_pil = st_cropper(pil_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    orig_img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    
    # --- ROUGHNESS CALCULATION PIPELINE ---
    st.subheader("2. Surface Topography & Texture Analysis")
    
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale values into a pseudo-height map matrix (scaled to physical microns)
    height_map = (gray.astype(np.float32) / 255.0) * intensity_scaling
    
    # Calculate global statistical benchmarks
    global_mu = np.mean(height_map)
    global_sa = np.mean(np.abs(height_map - global_mu))
    global_sq = np.sqrt(np.mean((height_map - global_mu) ** 2))
    
    # --- LOCALIZED ROUGHNESS MAPPING (Sliding Window RMS) ---
    # We use a moving standard deviation filter to map texture variation
    mean_kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
    local_mean = cv2.filter2D(height_map, -1, mean_kernel)
    local_sq_mean = cv2.filter2D(height_map ** 2, -1, mean_kernel)
    
    # Local Variance = E[X^2] - (E[X])^2
    local_variance = local_sq_mean - (local_mean ** 2)
    local_variance = np.clip(local_variance, 0, None) # Erase minor floating-point math artifacts
    local_sq_map = np.sqrt(local_variance) # Local Sq (Root Mean Square Roughness) Map
    
    # Normalize local roughness for heatmap generation
    roughness_normalized = cv2.normalize(local_sq_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(roughness_normalized, cv2.COLORMAP_JET)
    
    # Identify localized areas with abnormal roughness spikes (e.g., shattered struts or cracks)
    roughness_cutoff = global_sq * 1.5
    anomaly_mask = np.uint8(local_sq_map > roughness_cutoff) * 255
    
    # Generate overlay view
    overlay_display = orig_img.copy()
    overlay_display[anomaly_mask == 255] = [0, 0, 255] # Highlight highly irregular surfaces in solid Red
    
    # --- RENDER RESULTS PANEL ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_img, channels="BGR", caption="Selected Material Surface", use_container_width=True)
    with col2:
        st.image(heatmap, channels="BGR", caption="Roughness Heatmap (Blue=Smooth, Red=Rough)", use_container_width=True)
        
    st.subheader("📊 Quantitative Surface Profile Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Global Arithmetic Roughness (Sa)", 
            value=f"{global_sa:.2f} µm",
            help="Average absolute height deviation across the entire selected region."
        )
    with c2:
        st.metric(
            label="Global Root Mean Square Roughness (Sq)", 
            value=f"{global_sq:.2f} µm",
            help="Standard deviation of height profile. Highly responsive to extreme texture anomalies."
        )
    with c3:
        anomaly_area_pct = (np.count_nonzero(anomaly_mask) / anomaly_mask.size) * 100
        st.metric(
            label="Surface Texture Irregularity Area", 
            value=f"{anomaly_area_pct:.1f} %",
            help="Percentage of the scanned area showing a localized roughness score significantly higher than the baseline average."
        )
        
    with st.expander("Show Structural Roughness Breakdown"):
        st.image(overlay_display, channels="BGR", caption="Roughness Spike Locations (Marked in Red)", use_container_width=True)
        st.write("💡 *Interpretation:* In a healthy porous network, the roughness heatmap should remain fairly uniform. A clear linear red cluster passing through a strut wall indicates a macro structural anomaly or fracture path.")
else:
    st.info("Upload your multi-layered sample image to generate quantitative surface roughness statistics.")

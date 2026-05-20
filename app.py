import streamlit as st
import numpy as np
import cv2
from streamlit_cropper import st_cropper
from PIL import Image

# Page configuration
st.set_page_config(page_title="Linear Crack Analyzer", layout="centered")

st.title("🧽 High-Precision Linear Crack Detector")
st.write("Isolates straight, highly elongated structural cracks (Target Aspect Ratio: ~8) while filtering out ambient shadows.")

# Sidebar Controls for Calibration
st.sidebar.header("🔬 Target Geometric Filters")

target_aspect = st.sidebar.slider(
    "Target Aspect Ratio (Length/Width)", 
    min_value=3.0, 
    max_value=20.0, 
    value=8.0, 
    step=0.5,
    help="Target elongation of the crack. Structural splits typically rank much higher than pores or shadows."
)

aspect_tolerance = st.sidebar.slider(
    "Aspect Ratio Tolerance", 
    min_value=1.0, 
    max_value=10.0, 
    value=4.0, 
    step=0.5,
    help="Acceptable variance from target aspect ratio. (e.g. Target 8 with Tolerance 4 accepts ratios from 4 to 12)."
)

min_length = st.sidebar.slider(
    "Minimum Crack Length (Pixels)", 
    min_value=20, 
    max_value=500, 
    value=100, 
    step=10,
    help="Ensures minor localized micro-tears do not flag the system."
)

max_crack_width = st.sidebar.slider(
    "Max Crack Thickness Filter (Pixels)", 
    min_value=3, 
    max_value=51, 
    value=15, 
    step=2,
    help="Top-hat element width. Set this slightly wider than your target crack thickness to drop large shadows."
)

uploaded_file = st.file_uploader("Upload sponge surface image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    
    st.subheader("1. Select Analysis Area")
    st.info("Crop the active sponge area. Large global shadows outside the product boundary will be excluded.")
    
    # Crop implementation
    cropped_pil = st_cropper(pil_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    orig_img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    
    # --- PROCESSING PIPELINE ---
    st.subheader("2. Inspection Results")
    
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 1: Black Top-Hat Morphological Filter to suppress broad shadows
    # Isolate dark structures narrower than max_crack_width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_crack_width, max_crack_width))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    
    # Step 2: Otsu's adaptive thresholding on isolated narrow elements
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean isolated noise pixels
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    # Step 3: Extract contours for advanced structural shape description
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    validated_cracks_mask = np.zeros_like(binary)
    severe_crack_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:  # Base scale threshold to ignore noise
            continue
            
        # Compute oriented bounding box (handles any rotation angle safely)
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        
        # Prevent division by zero errors
        major_dim = max(w, h)
        minor_dim = max(1.0, min(w, h))
        
        calculated_aspect = major_dim / minor_dim
        
        # Calculate Extent (Straight-line density validation)
        # Straight cracks fill a box well; winding shadows have low box density
        box_area = major_dim * minor_dim
        extent = area / box_area if box_area > 0 else 0
        
        # GEOMETRIC BOUNDARY FILTERS
        # Check if calculated aspect ratio sits within selected tolerance bands
        is_correct_aspect = (target_aspect - aspect_tolerance) <= calculated_aspect <= (target_aspect + aspect_tolerance)
        is_long_enough = major_dim >= min_length
        is_straight = extent > 0.25  # Rejects highly erratic, winding shapes
        
        if is_correct_aspect and is_long_enough and is_straight:
            cv2.drawContours(validated_cracks_mask, [cnt], -1, 255, -1)
            severe_crack_count += 1

    # 4. Assessment display
    status = f"🔴 FAIL ({severe_crack_count} Structural Crack(s) Detected)" if severe_crack_count > 0 else "🟢 PASS"
    st.markdown(f"### Status: {status}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_img, channels="BGR", caption="Analyzed Sub-Region", use_container_width=True)
    with col2:
        overlay = orig_img.copy()
        overlay[validated_cracks_mask == 255] = [0, 0, 255]  # Paint verified features Red
        st.image(overlay, channels="BGR", caption="Targeted Linear Cracks Only", use_container_width=True)
        
    # Metrics Panel
    st.subheader("Structural Shape Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Verified Straight Cracks", value=severe_crack_count)
    with c2:
        total_px = validated_cracks_mask.size
        crack_px = np.count_nonzero(validated_cracks_mask)
        defect_pct = (crack_px / total_px) * 100
        st.metric(label="Calculated Linear Defect Area", value=f"{defect_pct:.3f} %")
        
    with st.expander("Show Advanced Computer Vision Diagnostic Maps"):
        col3, col4 = st.columns(2)
        with col3:
            st.image(tophat, caption="Top-Hat Filter (Shadows Removed)", use_container_width=True)
        with col4:
            st.image(binary, caption="Binarized Structural Map (Pre-Shape Filtering)", use_container_width=True)
else:
    st.info("Upload an image to execute high-aspect linear defect profiling.")

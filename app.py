import streamlit as st
import numpy as np
import cv2
from streamlit_cropper import st_cropper
from PIL import Image

# Page configuration
st.set_page_config(page_title="Deep Crack Detector", layout="centered")

st.title("🧽 Industrial Sponge: Deep & Long Crack Detector")
st.write("Select the region of interest (ROI) to exclude the background, then analyze for deep structural cracks.")

# Sidebar Controls for Filtering
st.sidebar.header("Target Defect Dimensions")

min_crack_length = st.sidebar.slider(
    "Minimum Crack Length (Pixels)", 
    min_value=10, 
    max_value=300, 
    value=80, 
    step=5,
    help="Filters out micro-tears. Higher values mean the app only catches longer cracks."
)

shadow_depth = st.sidebar.slider(
    "Shadow Depth (Darkness)", 
    min_value=10, 
    max_value=200, 
    value=45, 
    help="Lower values require cracks to be much darker (deeper shadow). Higher values allow lighter cracks."
)

uploaded_file = st.file_uploader("Upload sponge surface image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    # Open image with PIL for the cropper tool
    pil_image = Image.open(uploaded_file)
    
    st.subheader("1. Select Analysis Area")
    st.info("Drag the box corners below to select only the sponge material. Double-click or adjust to fit, then see the results below.")
    
    # Interactive Cropping Tool (Locks aspect ratio to Freeform so user can choose any shape)
    cropped_pil = st_cropper(pil_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    # Convert cropped PIL image to OpenCV format (BGR)
    orig_img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    
    # --- ANALYSIS STEP (Runs only on the cropped region) ---
    st.subheader("2. Inspection Results")
    
    # 1. Preprocessing
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Hard thresholding to capture only deep, dark shadows
    _, deep_voids = cv2.threshold(blurred, shadow_depth, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(deep_voids)
    
    validated_cracks = np.zeros_like(deep_voids)
    severe_crack_count = 0
    
    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        estimated_length = max(width, height)
        aspect_ratio = max(width, height) / max(1, min(width, height))
        
        # Filter: Long, large, and elongated
        if estimated_length >= min_crack_length and area > 30 and aspect_ratio > 1.5:
            validated_cracks[labels == i] = 255
            severe_crack_count += 1

    # 4. Calculate Pass/Fail
    status = "🔴 FAIL (Severe Crack Detected)" if severe_crack_count > 0 else "🟢 PASS"
    
    st.markdown(f"### Status: {status}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_img, channels="BGR", caption="Selected Sponge Region", use_container_width=True)
    with col2:
        # Create an overlay map (Paint validated deep cracks Red)
        overlay = orig_img.copy()
        overlay[validated_cracks == 255] = [0, 0, 255] # BGR color for Red
        st.image(overlay, channels="BGR", caption="Detected Deep Cracks", use_container_width=True)
        
    # Metrics
    st.subheader("Analysis Metrics")
    col3, col4 = st.columns(2)
    with col3:
        st.metric(label="Severe Cracks Found", value=severe_crack_count)
    with col4:
        total_pixels = validated_cracks.size
        crack_pixels = np.count_nonzero(validated_cracks)
        defect_percentage = (crack_pixels / total_pixels) * 100
        st.metric(label="Structural Damage Area", value=f"{defect_percentage:.2f} %")

else:
    st.info("Please upload an image of the sponge surface to run the crack check.")

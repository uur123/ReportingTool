import streamlit as st
import numpy as np
import cv2
from streamlit_cropper import st_cropper
from PIL import Image

# Page configuration
st.set_page_config(page_title="Hairline Strut Crack Detector", layout="centered")

st.title("🔬 Hairline Strut Crack Detector")
st.write("Specialized filter optimized to detect tight, thin fractures cutting across solid porous structures.")

# Sidebar Settings
st.sidebar.header("Crack Detection Adjustments")

crack_sensitivity = st.sidebar.slider(
    "Crack Sensitivity", 
    min_value=10, 
    max_value=100, 
    value=35,
    help="Lower values detect fainter, tighter cracks. Higher values prevent noise from triggering alarms."
)

min_crack_width_pixels = st.sidebar.slider(
    "Minimum Crack Length (Pixels)", 
    min_value=5, 
    max_value=100, 
    value=25,
    help="Filters out tiny pore textures. Only keeps continuous linear cracks."
)

uploaded_file = st.file_uploader("Upload sponge surface image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    
    st.subheader("1. Select Inspection Area")
    cropped_pil = st_cropper(pil_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    orig_img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    
    # --- ANALYSIS PIPELINE ---
    st.subheader("2. Crack Detection Output")
    
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Smooth out the fine micro-textures of the sponge wall without losing the crack line
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Step 2: Use Sobel derivative to find sharp horizontal transitions (Vertical changes)
    # This specifically targets lines slicing across the struts
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y)
    sobel_y = np.uint8(np.clip(sobel_y, 0, 255))
    
    # Step 3: Adaptive thresholding on the gradient map to isolate the sharpest crack lines
    # This bypasses the overall 3D shadows of the pores
    binary_cracks = cv2.adaptiveThreshold(
        sobel_y, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, - (105 - crack_sensitivity)
    )
    
    # Mask out the deep, dark open voids so we aren't scanning empty air space
    _, void_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    binary_cracks = cv2.bitwise_and(binary_cracks, cv2.bitwise_not(void_mask))
    
    # Step 4: Clean noise and group continuous lines
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)) # Horizontal alignment priority
    binary_cracks = cv2.morphologyEx(binary_cracks, cv2.MORPH_CLOSE, kernel_clean)
    
    # Step 5: Filter by size to ensure we only circle actual lines
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cracks)
    
    output_display = orig_img.copy()
    detected_cracks_count = 0
    final_crack_mask = np.zeros_like(binary_cracks)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # We look for features that are wider than they are tall (horizontal crack direction)
        # and meet our length requirement
        max_dimension = max(width, height)
        
        if max_dimension >= min_crack_width_pixels and area > 10:
            final_crack_mask[labels == i] = 255
            detected_cracks_count += 1
            
            # Draw a bounding rectangle box around the detected crack line
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(output_display, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 2)

    # UI Assessment Display
    status = f"🔴 FAIL ({detected_cracks_count} Hairline Fracture(s) Caught)" if detected_cracks_count > 0 else "🟢 PASS"
    st.markdown(f"### Inspection Assessment: {status}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_img, channels="BGR", caption="Input Sponge View", use_container_width=True)
    with col2:
        st.image(output_display, channels="BGR", caption="Detected Crack Locations (Red Boxes)", use_container_width=True)
        
    # Micro Diagnostic Views for Tuning
    with st.expander("Show Gradient Diagnostics"):
        col3, col4 = st.columns(2)
        with col3:
            st.image(sobel_y, caption="Step 1: Directional Gradient Map (Highlights Cross-Cuts)", use_container_width=True)
        with col4:
            st.image(binary_cracks, caption="Step 2: Isolated Threshold Lines", use_container_width=True)
else:
    st.info("Upload your sample image to test the updated gradient hairline detector.")

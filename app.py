import streamlit as st
import numpy as np
import cv2
from streamlit_cropper import st_cropper
from PIL import Image

# Page configuration
st.set_page_config(page_title="Foreground Roughness Analyzer", layout="centered")

st.title("🔬 Foreground Layer Surface Roughness Analyzer")
st.write("Isolates the crisp main foreground layer and computes roughness metrics exclusively on it, ignoring blurry deeper layers.")

# Sidebar Controls
st.sidebar.header("Foreground Layer Selection")

focus_threshold = st.sidebar.slider(
    "Sharpness Threshold (Focus Filter)", 
    min_value=5, 
    max_value=150, 
    value=35,
    help="Higher values select only the absolute crispest foreground surfaces. Lower values allow slightly deeper struts to blend in."
)

intensity_scaling = st.sidebar.slider(
    "Z-Axis Max Scale Calibration (µm)", 
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
    
    # --- FOREGROUND ISOLATION PIPELINE ---
    st.subheader("2. Foreground Segregation & Roughness Analysis")
    
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Calculate localized focus map using the Laplacian Operator (evaluates micro-edge sharpness)
    laplacian_map = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_variance = np.abs(laplacian_map)
    
    # Smooth the sharpness map to create uniform solid regions for the foreground struts
    focus_blurred = cv2.GaussianBlur(laplacian_variance, (11, 11), 0)
    
    # Create the Main Layer Mask (White = Main Layer, Black = Everything else)
    _, main_layer_mask = cv2.threshold(focus_blurred, focus_threshold, 255, cv2.THRESH_BINARY)
    main_layer_mask = main_layer_mask.astype(np.uint8)
    
    # Step 2: Extract Heights ONLY from the isolated Main Layer
    # Map pixels to physical heights
    height_map = (gray.astype(np.float32) / 255.0) * intensity_scaling
    
    # Extract height array values where the main layer mask is active (removes 2D structure bias)
    foreground_heights = height_map[main_layer_mask == 255]
    
    if len(foreground_heights) > 0:
        # Step 3: Compute Roughness parameters on the isolated foreground data array
        fg_mu = np.mean(foreground_heights)
        fg_sa = np.mean(np.abs(foreground_heights - fg_mu))
        fg_sq = np.sqrt(np.mean((foreground_heights - fg_mu) ** 2))
        
        # Step 4: Visual Generation
        # Build an extraction check map showing the isolated main layer inside your image
        main_layer_view = orig_img.copy()
        # Dim out the background layers to visually prove extraction isolation
        main_layer_view[main_layer_mask == 0] = (main_layer_view[main_layer_mask == 0] * 0.25).astype(np.uint8)
        
        # Build a true foreground-only height map for display visualization
        fg_height_display = np.zeros_like(height_map)
        fg_height_display[main_layer_mask == 255] = height_map[main_layer_mask == 255]
        fg_height_display_normalized = cv2.normalize(fg_height_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(fg_height_display_normalized, cv2.COLORMAP_JET)
        heatmap[main_layer_mask == 0] = 0 # Zero out background spaces in heatmap view
        
        # --- RENDER RESULTS PANEL ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(main_layer_view, channels="BGR", caption="Isolated Main Layer (Highlighted Surface)", use_container_width=True)
        with col2:
            st.image(heatmap, channels="BGR", caption="Main Layer Topography Heatmap", use_container_width=True)
            
        st.subheader("📊 Foreground Roughness Calculations")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Main Layer Roughness (Sa)", 
                value=f"{fg_sa:.2f} µm",
                help="Arithmetic mean height deviation restricted to the top in-focus strut surface."
            )
        with c2:
            st.metric(
                label="Main Layer Roughness (Sq)", 
                value=f"{fg_sq:.2f} µm",
                help="Root mean square height deviation restricted to the top in-focus strut surface."
            )
        with c3:
            surface_coverage = (np.count_nonzero(main_layer_mask) / main_layer_mask.size) * 100
            st.metric(
                label="Main Layer Density Space", 
                value=f"{surface_coverage:.1f} %",
                help="Percentage of image area occupied by the topmost physical sponge layer."
            )
    else:
        st.error("No foreground structural content isolated. Please lower the 'Sharpness Threshold' slider in the sidebar.")
        
    # Extra inspection views
    with st.expander("Show Computer Vision Depth Extraction Map"):
        st.image(focus_blurred, caption="Raw Local Edge Contrast Energy (Whiter = Closer to Camera)", use_container_width=True)
else:
    st.info("Upload your multi-layered sample view to run foreground layer segregation.")

import streamlit as st
import numpy as np
import cv2

# Page configuration
st.set_page_config(page_title="Sponge Crack Detector", layout="centered")

st.title("🧽 Sponge Crack Detection System")
st.write("Isolates and scores linear cracks in highly porous materials by filtering out natural pores.")

# Sidebar Controls
st.sidebar.header("Defect Sensitivity Settings")
kernel_size = st.sidebar.slider(
    "Pore Filter Size", 
    min_value=3, 
    max_value=31, 
    value=11, 
    step=2,
    help="Increase this value if natural pores are accidentally being flagged as cracks."
)

crack_threshold = st.sidebar.slider(
    "Darkness Sensitivity", 
    min_value=10, 
    max_value=150, 
    value=60, 
    help="Lower values catch only very dark/deep cracks. Higher values catch shallower cracks."
)

uploaded_file = st.file_uploader("Upload sponge surface image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    # Load Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    orig_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 1. Preprocessing
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    # Apply a light blur to smooth internal pore textures
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Adaptive Thresholding to extract ALL dark spaces (Pores + Cracks)
    # Binary inverse: Shadows become White (255), surface becomes Black (0)
    all_voids = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 51, crack_threshold - 50
    )
    
    # 3. Morphological Filtering to Separate Pores from Cracks
    # Create a circular kernel to mimic natural pore shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 'Opening' eliminates features smaller/narrower than the kernel (erases fine cracks)
    only_pores = cv2.morphologyEx(all_voids, cv2.MORPH_OPEN, kernel)
    
    # Subtract pores from the total voids map to isolate the long, narrow cracks
    only_cracks = cv2.subtract(all_voids, only_pores)
    
    # Clean up random single-pixel noise from the final crack map
    only_cracks = cv2.morphologyEx(only_cracks, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    # 4. Calculate Defect Score
    total_pixels = only_cracks.size
    crack_pixels = np.count_nonzero(only_cracks)
    defect_percentage = (crack_pixels / total_pixels) * 100
    
    # Define simple pass/fail status
    status = "🔴 FAIL (Crack Detected)" if defect_percentage > 0.5 else "🟢 PASS"
    
    # Display Visualizations Side-by-Side
    st.header(f"Inspection Result: {status}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_img, channels="BGR", caption="Original Sponge Surface", use_container_width=True)
    with col2:
        # Create an overlay map (Red highlights where cracks are detected)
        overlay = orig_img.copy()
        overlay[only_cracks == 255] = [0, 0, 255] # Paint crack pixels red
        st.image(overlay, channels="BGR", caption="Detected Cracks (Red Overlay)", use_container_width=True)
        
    # Metrics
    st.subheader("Analysis Metrics")
    st.metric(
        label="Crack Density Score", 
        value=f"{defect_percentage:.2f} %",
        help="Percentage of the visible surface area identified as a crack defect."
    )
    
    # Diagnostic views for debugging lighting/pores
    with st.expander("See Diagnostic Binary Maps"):
        col3, col4 = st.columns(2)
        with col3:
            st.image(all_voids, caption="Step 1: All Voids Detected", use_container_width=True)
        with col4:
            st.image(only_pores, caption="Step 2: Natural Pores Isolated", use_container_width=True)

else:
    st.info("Please upload an image of the sponge surface to run the crack check.")

import streamlit as st
import numpy as np
import cv2

# Page configuration
st.set_page_config(page_title="Image Roughness Analyzer", layout="centered")

st.title("🔬 Image Surface Roughness Analyzer")
st.write("Upload an image to estimate surface roughness ($S_a$ and $S_q$) based on pixel intensity variations.")

# Sidebar Controls
st.sidebar.header("Calibration Settings")
# Conversion factor: How many micrometers does one pixel represent horizontally/vertically?
microns_per_pixel = st.sidebar.slider(
    "Pixel Scale (µm per pixel)", 
    min_value=0.1, 
    max_value=100.0, 
    value=1.0, 
    step=0.1,
    help="Horizontal calibration factor to map pixels to physical distance."
)

# Height scale factor: maps grayscale (0-255) to a physical height range in microns
height_scale = st.sidebar.slider(
    "Max Peak Height Mapping (µm)", 
    min_value=1.0, 
    max_value=1000.0, 
    value=50.0, 
    step=1.0,
    help="Maps the brightest white pixel (255) to this physical height value in micrometers."
)

# File Uploader
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    # Read image file buffer into bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)
    
    # 1. Preprocessing
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to filter out microscopic high-frequency sensor noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Map Pixels to Physical Height Matrix Z(x,y)
    # Normalize 0-255 scale to 0.0-1.0, then scale to maximum physical height input
    Z = (blurred / 255.0) * height_scale
    
    # Get matrix dimensions
    M, N = Z.shape
    
    with st.spinner("Calculating roughness metrics..."):
        # Calculate mean height plane (μ)
        mean_height = np.mean(Z)
        
        # Calculate Arithmetic Mean Roughness (Sa)
        sa = np.mean(np.abs(Z - mean_height))
        
        # Calculate Root Mean Square Roughness (Sq)
        sq = np.sqrt(np.mean((Z - mean_height) ** 2))
    
    # Display results
    st.header("📊 Roughness Scores")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Arithmetic Mean Roughness (Sa)", 
            value=f"{sa:.4f} µm",
            help="Average absolute deviation from the mean plane surface height."
        )
    with col2:
        st.metric(
            label="Root Mean Square Roughness (Sq)", 
            value=f"{sq:.4f} µm",
            help="Standard deviation of the surface height profile. Highly sensitive to extreme peaks and valleys."
        )
        
    # Additional Context Metrics
    st.subheader("Surface Dimensions")
    physical_width = N * microns_per_pixel
    physical_height = M * microns_per_pixel
    st.write(f"**Analyzed Resolution:** {N} x {M} pixels")
    st.write(f"**Physical Area Evaluated:** {physical_width:.2f} µm x {physical_height:.2f} µm")

else:
    st.info("Please upload an image file to begin calculations.")

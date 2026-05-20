import streamlit as st
import numpy as np
import cv2
from streamlit_cropper import st_cropper
from PIL import Image

# Page configuration
st.set_page_config(page_title="Strut Crack Detector", layout="centered")

st.title("🧽 Sponge Strut Micro-Crack Detector")
st.write("Identifies tiny 90-degree gaps and snapped structural struts inside open-cell porous foam materials.")

# Sidebar Settings
st.sidebar.header("Micro-Crack Sensitivity")

material_brightness = st.sidebar.slider(
    "Sponge Material Threshold", 
    min_value=30, 
    max_value=230, 
    value=120,
    help="Adjust this until only the solid sponge struts are highlighted as white, and pores are pitch black."
)

max_gap_size = st.sidebar.slider(
    "Max Crack Gap (Pixels)", 
    min_value=2, 
    max_value=20, 
    value=8,
    help="The maximum width of the micro-gap between a broken strut. Keeps natural open pores from triggering alarms."
)

uploaded_file = st.file_uploader("Upload micro-structure image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    
    st.subheader("1. Select Target Strut Area")
    cropped_pil = st_cropper(pil_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    orig_img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    
    # --- PIPELINE TO DETECT BROKEN STRUTS ---
    st.subheader("2. Structural Analysis")
    
    # Convert and threshold to isolate ONLY the solid struts (solid = white, pores = black)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, solid_struts = cv2.threshold(blurred, material_brightness, 255, cv2.THRESH_BINARY)
    
    # Clean up micro-noise inside the solid material
    solid_struts = cv2.morphologyEx(solid_struts, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    # Skeletonization: Reduce struts to a 1-pixel-thick line representation
    # This exposes the "topology" of the sponge mesh grid
    skeleton = cv2.ximgproc.thinning(solid_struts, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    # Identify Endpoints (where a line just ends without hitting a junction node)
    # Define Hit-or-Miss lookup kernels to find line endings in 8 directions
    endpoints_mask = np.zeros_like(skeleton)
    
    # Basic lookup kernels for endpoints (1 pixel surrounded by zeros)
    kernels = [
        np.array([[-1, -1, -1], [-1,  1, -1], [-1,  1, -1]]), # Vertical bottom-end
        np.array([[-1,  1, -1], [-1,  1, -1], [-1, -1, -1]]), # Vertical top-end
        np.array([[-1, -1, -1], [-1,  1,  1], [-1, -1, -1]]), # Horizontal right-end
        np.array([[-1, -1, -1], [ 1,  1, -1], [-1, -1, -1]]), # Horizontal left-end
        np.array([[ 1, -1, -1], [-1,  1, -1], [-1, -1, -1]]), # Diagonal top-left
        np.array([[-1, -1,  1], [-1,  1, -1], [-1, -1, -1]]), # Diagonal top-right
        np.array([[-1, -1, -1], [-1,  1, -1], [ 1, -1, -1]]), # Diagonal bottom-left
        np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1,  1]])  # Diagonal bottom-right
    ]
    
    for k in kernels:
        hit_miss = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, k)
        endpoints_mask = cv2.bitwise_or(endpoints_mask, hit_miss)
        
    # Pair Matching: Group endpoints facing each other across a small split gap
    # Dilate endpoints slightly to see if opposing broken ends touch each other
    dilated_endpoints = cv2.dilate(endpoints_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (max_gap_size, max_gap_size)))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_endpoints)
    
    # Draw tracking markers on output display image
    output_display = orig_img.copy()
    verified_crack_count = 0
    
    for i in range(1, num_labels):
        # Count how many individual endpoint points fall inside this small neighborhood box
        # If a single strut snapped, it produces exactly 2 endpoints facing one another
        pixels_in_component = endpoints_mask[labels == i]
        endpoint_count = np.count_nonzero(pixels_in_component)
        
        if endpoint_count >= 2:
            verified_crack_count += 1
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            # Draw an indicator target over the broken strut location
            cv2.circle(output_display, (cx, cy), 12, (0, 0, 255), 2)
            cv2.circle(output_display, (cx, cy), 2, (0, 0, 255), -1)

    # Output Rendering
    status = f"🔴 FAIL ({verified_crack_count} Snapped Strut Cracks)" if verified_crack_count > 0 else "🟢 PASS"
    st.markdown(f"### Inspection Assessment: {status}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_img, channels="BGR", caption="Input Sponge View", use_container_width=True)
    with col2:
        st.image(output_display, channels="BGR", caption="Targeted Broken Struts (Red Circles)", use_container_width=True)
        
    # Micro Diagnostic Views
    with st.expander("Show Microstructural Extraction Steps"):
        col3, col4 = st.columns(2)
        with col3:
            st.image(solid_struts, caption="Isolated Solid Struts (White Matrix)", use_container_width=True)
        with col4:
            # Upscale skeleton line slightly so it's easily visible in the UI
            visible_skeleton = cv2.dilate(skeleton, np.ones((2,2), np.uint8))
            st.image(visible_skeleton, caption="1-Pixel Skeleton Grid Model", use_container_width=True)
else:
    st.info("Upload a clear macro/micro image of the sponge pores to scan for broken struts.")

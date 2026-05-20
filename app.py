import streamlit as st
import numpy as np
import cv2
from streamlit_cropper import st_cropper
from PIL import Image

# Page configuration
st.set_page_config(page_title="Multi-Layer Crack Filter", layout="centered")

st.title("🧽 Multi-Layered Sponge Vector Crack Detector")
st.write("Isolates deep fractures by filtering out chaotic background layers using localized directional texture tuning.")

# Sidebar Settings
st.sidebar.header("Gabor Wave Vector Tuning")

crack_angle = st.sidebar.slider(
    "Target Crack Angle (Degrees)", 
    min_value=0, 
    max_value=180, 
    value=0,
    help="0° targets horizontal cracks. 90° targets vertical cracks. Adjust to match the orientation of the crack."
)

crack_thickness = st.sidebar.slider(
    "Crack Thickness Scale (Wavelength)", 
    min_value=2.0, 
    max_value=30.0, 
    value=6.0, 
    step=0.5,
    help="Matches the pixel thickness of the hairline crack. Smaller numbers catch tighter tears."
)

anomaly_threshold = st.sidebar.slider(
    "Detection Sensitivity Threshold", 
    min_value=10, 
    max_value=255, 
    value=130,
    help="Lower values find subtle aligned micro-cracks. Higher values ensure only heavy fractures break through."
)

uploaded_file = st.file_uploader("Upload multi-layered sponge image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    
    st.subheader("1. Select Inspection Area")
    cropped_pil = st_cropper(pil_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    orig_img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    
    # --- GABOR DIRECTIONAL VECTOR PIPELINE ---
    st.subheader("2. Directional Texture Response")
    
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # Construct a Gabor Filter tailored to catch lines at a specific angle
    # theta: Orientation angle converted to radians
    theta = np.deg2rad(crack_angle)
    # sigma: Gaussian envelope bandwidth size
    sigma = crack_thickness * 0.5
    # lambda: Wavelength of the sinusoidal factor
    lambd = crack_thickness
    # gamma: Spatial aspect ratio (0.5 forces elongation along the target line)
    gamma = 0.5
    # psi: Phase offset
    psi = 0
    
    gabor_kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    
    # Run the directional filter over the multi-layer image
    gabor_response = cv2.filter2D(gray, cv2.CV_8U, gabor_kernel)
    
    # Enhance local contrast to pull the crack out of depth shadows
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_response = clahe.apply(gabor_response)
    
    # Isolate strong directional alignment peaks
    _, binary_anomalies = cv2.threshold(enhanced_response, anomaly_threshold, 255, cv2.THRESH_BINARY)
    
    # Link nearby fragments along the horizontal crack path using morphological closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3) if crack_angle in range(0, 45) or crack_angle in range(135, 181) else (3, 15))
    linked_cracks = cv2.morphologyEx(binary_anomalies, cv2.MORPH_CLOSE, kernel_close)
    
    # Filter out remaining non-linear background noise fragments using contour analysis
    contours, _ = cv2.findContours(linked_cracks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_display = orig_img.copy()
    crack_indicators_count = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        max_dim = max(w, h)
        
        # Real cracks spanning across structural components will have an expanded bounding length
        if max_dim >= 20:
            crack_indicators_count += 1
            cv2.rectangle(output_display, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 2)
            cv2.putText(output_display, "VECTOR DEFECT", (x, y - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # UI Assessment Display
    status = f"🔴 FAIL ({crack_indicators_count} Aligned Fractures Isolated)" if crack_indicators_count > 0 else "🟢 PASS"
    st.markdown(f"### Inspection Assessment: {status}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_img, channels="BGR", caption="Input Multi-Layer Image", use_container_width=True)
    with col2:
        st.image(output_display, channels="BGR", caption="Isolated Linear Anomalies (Red Boxes)", use_container_width=True)
        
    # Multi-Layer Diagnostics
    with st.expander("Show Directional Vector Diagnostic Maps"):
        col3, col4 = st.columns(2)
        with col3:
            st.image(enhanced_response, caption="Gabor Filter Output (Isolates Aligned Waves)", use_container_width=True)
        with col4:
            st.image(linked_cracks, caption="Binarized Linear Wave Patterns", use_container_width=True)
else:
    st.info("Upload your multi-layered sample image to execute anisotropic frequency filtering.")

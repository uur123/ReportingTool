import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from PIL import Image

from utils.image_processing import (
    preprocess_image,
    segment_pores,
    calculate_porosity,
    pore_analysis,
    detect_cracks,
    crack_analysis,
    local_thickness_map
)

from utils.texture_analysis import texture_features
from utils.fractal import fractal_dimension
from utils.metrics import structural_weakness_index

st.set_page_config(layout="wide")

st.title("Porous Ceramic Structural Integrity Analyzer")

st.markdown("""
This application performs structural, pore, and crack morphological tracking.
It features a focus-plane layer isolation tool designed for open-cell sponge matrices.
""")

uploaded_files = st.file_uploader(
    "Upload microscopy images",
    type=["png", "jpg", "jpeg", "tif"],
    accept_multiple_files=True
)

results = []

if uploaded_files:
    st.sidebar.header("1. General Processing")
    blur_size = st.sidebar.slider("Gaussian Blur Radius", 1, 15, 5, step=2)
    threshold = st.sidebar.slider("Pore Segmentation Threshold", 0, 255, 120)

    # --- NEW SPONGE FOCUS LAYER FILTER CONTROLS ---
    st.sidebar.header("2. Sponge 3D Layer Isolator")
    top_layer_focus = st.sidebar.checkbox("Focus Only on Top Layer", value=True)
    focus_sensitivity = st.sidebar.slider("Focus Filter Sensitivity", 1, 100, 15, 
                                          help="Lower ignores more depth blur. Higher allows deeper layers.")

    st.sidebar.header("3. Strut Zone Calibration")
    strut_threshold = st.sidebar.slider("Strut Grey-level Threshold", 0, 255, 40)
    strut_invert = st.sidebar.checkbox("Invert Strut Logic (Check if struts are dark)", value=False)

    st.sidebar.header("4. Crack Filter Tweaking")
    crack_polarity = st.sidebar.selectbox("Crack Contrast Type", ["Dark Cracks", "Bright Cracks"])
    crack_threshold = st.sidebar.slider("Crack Sensitivity (Hessian Cutoff)", 1, 200, 40)
    crack_sigma = st.sidebar.slider("Target Crack Width (Sigma)", 0.5, 5.0, 1.0, step=0.1)
    min_eccentricity = st.sidebar.slider("Minimum Elongation (Eccentricity)", 0.50, 0.99, 0.75, step=0.05)
    min_crack_size = st.sidebar.slider("Minimum Crack Size (Pixels)", 1, 50, 3)

    for file in uploaded_files:
        st.header(file.name)

        image = Image.open(file)
        image = np.array(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        processed = preprocess_image(gray, blur_size)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(gray, use_container_width=True)
        with col2:
            st.subheader("Processed")
            st.image(processed, use_container_width=True)

        pore_mask = segment_pores(processed, threshold)

        # Execute surface-focused crack detection
        crack_mask = detect_cracks(
            processed, 
            threshold=crack_threshold, 
            pore_mask=pore_mask,
            strut_threshold=strut_threshold,
            strut_invert=strut_invert,
            crack_polarity=crack_polarity,
            crack_sigma=crack_sigma,
            min_eccentricity=min_eccentricity,
            min_crack_size=min_crack_size,
            top_layer_focus=top_layer_focus,
            focus_sensitivity=focus_sensitivity
        )

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Pore Segmentation Mask")
            st.image(pore_mask, use_container_width=True)
        with col4:
            st.subheader("Crack Segmentation Mask")
            st.image(crack_mask, use_container_width=True)

        # Metric Analytics Calculations
        porosity = calculate_porosity(pore_mask)
        pore_data = pore_analysis(pore_mask)
        crack_data = crack_analysis(crack_mask)
        texture = texture_features(processed)
        fractal = fractal_dimension(pore_mask)
        thickness_map, avg_thickness = local_thickness_map(pore_mask)

        swi = structural_weakness_index(
            porosity,
            crack_data["density"],
            crack_data["mean_length"]
        )

        metrics = {
            "sample": file.name,
            "porosity_percent": porosity,
            "mean_pore_area": pore_data["mean_area"],
            "pore_count": pore_data["count"],
            "mean_crack_length": crack_data["mean_length"],
            "max_crack_length": crack_data["max_length"],
            "crack_count": crack_data["count"],
            "crack_density": crack_data["density"],
            "mean_aspect_ratio": crack_data["mean_aspect_ratio"],
            "fractal_dimension": fractal,
            "texture_entropy": texture["entropy"],
            "glcm_contrast": texture["contrast"],
            "avg_thickness": avg_thickness,
            "structural_weakness_index": swi
        }

        results.append(metrics)

        st.subheader("Metrics Summary")
        st.dataframe(pd.DataFrame([metrics]))

        st.subheader("Pore Size Distribution")
        fig1, ax1 = plt.subplots()
        ax1.hist(pore_data["areas"], bins=20 if len(pore_data["areas"]) > 0 else 5)
        ax1.set_xlabel("Pore Area")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        st.subheader("Crack Length Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(crack_data["lengths"], bins=20 if len(crack_data["lengths"]) > 0 else 5)
        ax2.set_xlabel("Crack Length")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

        st.subheader("Local Thickness Map")
        fig3, ax3 = plt.subplots()
        im = ax3.imshow(thickness_map, cmap="viridis")
        plt.colorbar(im)
        st.pyplot(fig3)

    if len(results) > 1:
        st.header("Batch Comparison")
        comparison_df = pd.DataFrame(results)
        st.dataframe(comparison_df)

        fig4 = px.bar(
            comparison_df,
            x="sample",
            y="structural_weakness_index",
            title="Structural Weakness Index Comparison"
        )
        st.plotly_chart(fig4, use_container_width=True)

        csv = comparison_df.to_csv(index=False)
        st.download_button("Download CSV Summary", csv, "analysis_results.csv", "text/csv")

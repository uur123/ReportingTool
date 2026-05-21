
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
This application performs:

- Porosity analysis
- Pore size distribution
- Crack detection
- Crack morphology analysis
- Texture characterization
- Fractal dimension estimation
- Structural weakness estimation
""")

uploaded_files = st.file_uploader(
    "Upload microscopy images",
    type=["png", "jpg", "jpeg", "tif"],
    accept_multiple_files=True
)

results = []

if uploaded_files:

    blur_size = st.sidebar.slider("Gaussian Blur", 1, 15, 5, step=2)
    threshold = st.sidebar.slider("Pore Threshold", 0, 255, 120)
    crack_threshold = st.sidebar.slider("Crack Threshold", 0, 255, 80)

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

        crack_mask = detect_cracks(processed, crack_threshold)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Pore Segmentation")
            st.image(pore_mask, use_container_width=True)

        with col4:
            st.subheader("Crack Segmentation")
            st.image(crack_mask, use_container_width=True)

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

        st.subheader("Metrics")

        st.dataframe(pd.DataFrame([metrics]))

        st.subheader("Pore Size Distribution")

        fig1, ax1 = plt.subplots()

        ax1.hist(pore_data["areas"], bins=20)

        ax1.set_xlabel("Pore Area")
        ax1.set_ylabel("Frequency")

        st.pyplot(fig1)

        st.subheader("Crack Length Distribution")

        fig2, ax2 = plt.subplots()

        ax2.hist(crack_data["lengths"], bins=20)

        ax2.set_xlabel("Crack Length")
        ax2.set_ylabel("Frequency")

        st.pyplot(fig2)

        st.subheader("Local Thickness Map")

        fig3, ax3 = plt.subplots()

        im = ax3.imshow(thickness_map, cmap="viridis")

        plt.colorbar(im)

        st.pyplot(fig3)

    if len(results) > 1:

        st.header("Comparison")

        comparison_df = pd.DataFrame(results)

        st.dataframe(comparison_df)

        fig4 = px.bar(
            comparison_df,
            x="sample",
            y="structural_weakness_index",
            title="Structural Weakness Index"
        )

        st.plotly_chart(fig4, use_container_width=True)

        csv = comparison_df.to_csv(index=False)

        st.download_button(
            "Download CSV",
            csv,
            "analysis_results.csv",
            "text/csv"
        )

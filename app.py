import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from utils.image_processing import (
    preprocess_image,
    segment_image,
    calculate_porosity,
    pore_analysis
)

from utils.texture_analysis import texture_features
from utils.fractal import fractal_dimension
from utils.metrics import compare_samples


st.set_page_config(layout="wide")

st.title("Microscopy-Based Surface Porosity Analysis")

st.markdown("""
This app performs:
- Surface porosity analysis
- Texture characterization
- Pore distribution analysis
- Fractal roughness estimation
""")

uploaded_files = st.file_uploader(
    "Upload microscopy images",
    type=["png", "jpg", "jpeg", "tif"],
    accept_multiple_files=True
)

results = []

if uploaded_files:

    for file in uploaded_files:

        st.header(file.name)

        image = Image.open(file)
        image = np.array(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        st.subheader("Original Image")
        st.image(gray, use_container_width=True)

        blur_size = st.slider(
            f"Gaussian Blur Kernel - {file.name}",
            1,
            15,
            5,
            step=2
        )

        threshold = st.slider(
            f"Threshold - {file.name}",
            0,
            255,
            120
        )

        processed = preprocess_image(gray, blur_size)

        segmented = segment_image(processed, threshold)

        st.subheader("Segmented Image")
        st.image(segmented, use_container_width=True)

        porosity = calculate_porosity(segmented)

        pore_data = pore_analysis(segmented)

        texture = texture_features(processed)

        fractal = fractal_dimension(segmented)

        metrics = {
            "sample": file.name,
            "porosity_percent": porosity,
            "mean_pore_area": pore_data["mean_area"],
            "pore_count": pore_data["count"],
            "glcm_contrast": texture["contrast"],
            "glcm_homogeneity": texture["homogeneity"],
            "glcm_energy": texture["energy"],
            "entropy": texture["entropy"],
            "fractal_dimension": fractal
        }

        results.append(metrics)

        st.subheader("Metrics")

        df_single = pd.DataFrame([metrics])

        st.dataframe(df_single)

        st.subheader("Pore Size Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            pore_data["areas"],
            bins=20
        )

        ax.set_xlabel("Pore Area")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    if len(results) > 1:

        st.header("Sample Comparison")

        comparison_df = pd.DataFrame(results)

        st.dataframe(comparison_df)

        st.subheader("Porosity Comparison")

        fig2, ax2 = plt.subplots()

        ax2.bar(
            comparison_df["sample"],
            comparison_df["porosity_percent"]
        )

        ax2.set_ylabel("Porosity (%)")

        plt.xticks(rotation=45)

        st.pyplot(fig2)

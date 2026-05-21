import cv2
import numpy as np
from skimage import measure


def preprocess_image(image, blur_size=5):

    processed = cv2.GaussianBlur(
        image,
        (blur_size, blur_size),
        0
    )

    processed = cv2.equalizeHist(processed)

    return processed


def segment_image(image, threshold=120):

    _, segmented = cv2.threshold(
        image,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    return segmented


def calculate_porosity(segmented):

    void_pixels = np.sum(segmented > 0)

    total_pixels = segmented.size

    porosity = (void_pixels / total_pixels) * 100

    return round(porosity, 2)


def pore_analysis(segmented):

    labels = measure.label(segmented)

    props = measure.regionprops(labels)

    areas = []

    for prop in props:

        if prop.area > 10:
            areas.append(prop.area)

    if len(areas) == 0:
        mean_area = 0
    else:
        mean_area = np.mean(areas)

    return {
        "areas": areas,
        "mean_area": mean_area,
        "count": len(areas)
    }

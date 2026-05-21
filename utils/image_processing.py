
import cv2
import numpy as np

from scipy.ndimage import distance_transform_edt

from skimage import measure
from skimage.morphology import skeletonize


def preprocess_image(image, blur_size=5):

    img = cv2.GaussianBlur(
        image,
        (blur_size, blur_size),
        0
    )

    img = cv2.equalizeHist(img)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    img = clahe.apply(img)

    return img


def segment_pores(image, threshold=120):

    _, mask = cv2.threshold(
        image,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


def calculate_porosity(mask):

    pore_pixels = np.sum(mask > 0)

    total_pixels = mask.size

    return round((pore_pixels / total_pixels) * 100, 2)


def pore_analysis(mask):

    labels = measure.label(mask)

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


def detect_cracks(image, threshold=80):

    edges = cv2.Canny(
        image,
        threshold,
        threshold * 2
    )

    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def crack_analysis(mask):

    binary = mask > 0

    skeleton = skeletonize(binary)

    labels = measure.label(skeleton)

    props = measure.regionprops(labels)

    lengths = []
    aspect_ratios = []

    total_area = mask.shape[0] * mask.shape[1]

    for prop in props:

        if prop.area > 5:

            lengths.append(prop.area)

            major = prop.major_axis_length
            minor = prop.minor_axis_length

            if minor == 0:
                ar = 0
            else:
                ar = major / minor

            aspect_ratios.append(ar)

    if len(lengths) == 0:

        mean_length = 0
        max_length = 0
        mean_ar = 0

    else:

        mean_length = np.mean(lengths)
        max_length = np.max(lengths)
        mean_ar = np.mean(aspect_ratios)

    density = len(lengths) / total_area

    return {
        "lengths": lengths,
        "mean_length": mean_length,
        "max_length": max_length,
        "mean_aspect_ratio": mean_ar,
        "count": len(lengths),
        "density": density
    }


def local_thickness_map(mask):

    binary = mask > 0

    distance = distance_transform_edt(binary)

    avg_thickness = np.mean(distance[binary]) * 2

    return distance, avg_thickness

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


def detect_cracks(image, threshold=80, pore_mask=None):
    """
    Detects cracks by extracting edges and applying shape-based filters 
    to remove circular or solid pore artifacts.
    """
    # 1. Base edge detection
    edges = cv2.Canny(
        image,
        threshold,
        threshold * 2
    )

    # 2. Subtract the pore regions so we only focus on the ceramic struts
    if pore_mask is not None:
        # Dilate pore mask slightly to ensure pore boundaries are entirely removed
        kernel_pore = np.ones((3, 3), np.uint8)
        dilated_pores = cv2.dilate(pore_mask, kernel_pore, iterations=1)
        edges = cv2.bitwise_and(edges, cv2.bitwise_not(dilated_pores))

    # 3. Morphological closing to bridge small gaps in the cracks
    kernel_edges = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel_edges, iterations=1)

    # 4. Filter connected components based on crack geometry (long and thin)
    labeled_edges = measure.label(edges > 0)
    props = measure.regionprops(labeled_edges)
    
    cleaned_crack_mask = np.zeros_like(edges)

    for prop in props:
        # Ignore tiny noise artifacts
        if prop.area < 5:
            continue
            
        # Eccentricity ranges from 0 (perfect circle) to 1 (straight line)
        # Solidity checks how compact the shape is (pores = high, cracks = low)
        if prop.eccentricity > 0.85 and prop.solidity < 0.6:
            # Reconstruct the validated crack back into our binary mask
            for coord in prop.coords:
                cleaned_crack_mask[coord[0], coord[1]] = 255

    return cleaned_crack_mask


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

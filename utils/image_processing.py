import cv2
import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.morphology import skeletonize
# New imports for ridge and line detection without edges
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


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
    Detects internal cracks within ceramic struts using Hessian Ridge Analysis.
    Ignores structural boundaries and pores completely.
    """
    # 1. Segment the Ceramic Struts (Assume struts are the lighter/mid-tone material)
    # Adjust this if your struts are darker than the background
    _, strut_mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    
    # 2. Isolate internal strut body by removing pores and eroding boundaries
    if pore_mask is not None:
        strut_body = cv2.bitwise_and(strut_mask, cv2.bitwise_not(pore_mask))
    else:
        strut_body = strut_mask.copy()
        
    # Erode the strut mask to pull away from outer edges/boundaries
    kernel_erode = np.ones((5, 5), np.uint8)
    safe_strut_zone = cv2.erode(strut_body, kernel_erode, iterations=2)

    # 3. Use Hessian Matrix Eigenvalues to find thin line valleys (ridges) instead of Canny edges
    # This prevents catching boundaries and looks for thin, linear dark structures
    hxx, hxy, hyy = hessian_matrix(image, sigma=1.5)
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    
    # i1 represents the maximal local curvature. Cracks show up as strong positive/negative peaks.
    # We look for dark line structures on a lighter strut background
    ridge_map = np.where(i1 > (threshold / 10.0), 255, 0).astype(np.uint8)

    # 4. Strictly constrain cracks to the inside of the safe strut zones
    valid_cracks = cv2.bitwise_and(ridge_map, safe_strut_zone)

    # 5. Filter out remaining non-elongated noise
    labeled_cracks = measure.label(valid_cracks > 0)
    props = measure.regionprops(labeled_cracks)
    
    final_crack_mask = np.zeros_like(valid_cracks)
    for prop in props:
        if prop.area < 8: # Filter out tiny noise dots
            continue
        # Only keep elongated features (cracks)
        if prop.eccentricity > 0.88:
            for coord in prop.coords:
                final_crack_mask[coord[0], coord[1]] = 255

    return final_crack_mask


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
            ar = major / minor if minor != 0 else 0
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

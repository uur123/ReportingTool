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
    Diagnostic-enabled Hessian crack detector. 
    Allows tuning for dark-on-light or light-on-dark crack structures.
    """
    # 1. Segment the Ceramic Struts (Ensure threshold matches your strut color)
    # If your struts are dark, change cv2.THRESH_BINARY to cv2.THRESH_BINARY_INV
    _, strut_mask = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
    
    # 2. Build the Safe Strut Zone
    if pore_mask is not None:
        strut_body = cv2.bitwise_and(strut_mask, cv2.bitwise_not(pore_mask))
    else:
        strut_body = strut_mask.copy()
        
    # Keep erosion minimal (3x3 instead of 5x5) so we don't accidentally erase narrow struts
    kernel_erode = np.ones((3, 3), np.uint8)
    safe_strut_zone = cv2.erode(strut_body, kernel_erode, iterations=1)

    # 3. Calculate Hessian Matrix
    # Lower sigma (e.g., 1.0) catches thinner, sharper cracks
    hxx, hxy, hyy = hessian_matrix(image, sigma=1.0)
    i1, i2 = hessian_matrix_eigvals([hxx, hxy, hyy])
    
    # --- CONTRAST SELECTION ---
    # OPTION A: Use 'i1' if your cracks are DARK lines on a LIGHT material background.
    # OPTION B: Use 'i2' if your cracks are LIGHT lines on a DARK material background.
    # Change 'i1' below to 'i2' if your crack lines are bright/white.
    raw_intensity = i1 

    # Convert threshold to fit Hessian scale. Lowering this value captures fainter lines.
    hessian_cutoff = threshold / 50.0
    ridge_map = np.where(raw_intensity > hessian_cutoff, 255, 0).astype(np.uint8)

    # 4. Restrict features to the interior of our struts
    valid_cracks = cv2.bitwise_and(ridge_map, safe_strut_zone)

    # 5. Geometrical filtering
    labeled_cracks = measure.label(valid_cracks > 0)
    props = measure.regionprops(labeled_cracks)
    
    final_crack_mask = np.zeros_like(valid_cracks)
    
    # Terminal debug print statement to find out why shapes are being discarded
    print(f"[DEBUG] Found {len(props)} total raw ridge segments before shape filtering.")

    for prop in props:
        # Lowered size limit to 3 pixels so short/faint cracks are not deleted
        if prop.area < 3: 
            continue
            
        # RELAXED RULES: Lowered required eccentricity from 0.88 to 0.75 
        # Left solidity check open so branching/crooked cracks pass through safely
        if prop.eccentricity > 0.75:
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

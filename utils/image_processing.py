import cv2
import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.morphology import skeletonize
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def preprocess_image(image, blur_size=5):
    if blur_size % 2 == 0:
        blur_size += 1
    img = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    img = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


def segment_pores(image, threshold=120):
    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def calculate_porosity(mask):
    pore_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    if total_pixels == 0:
        return 0.0
    return round((pore_pixels / total_pixels) * 100, 2)


def pore_analysis(mask):
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    areas = []
    for prop in props:
        if prop.area > 10:
            areas.append(prop.area)
    mean_area = np.mean(areas) if len(areas) > 0 else 0
    return {"areas": areas, "mean_area": mean_area, "count": len(areas)}


def detect_cracks(
    image, 
    threshold=80, 
    pore_mask=None, 
    strut_threshold=40,
    strut_invert=False,
    crack_polarity="Dark Cracks",
    crack_sigma=1.0,
    min_eccentricity=0.75,
    min_crack_size=3,
    top_layer_focus=True,
    focus_sensitivity=15
):
    """
    Advanced Hessian-based crack detector with an automated 
    focal layer isolator for 3D sponge/foam strut networks.
    """
    # 1. OPTIONAL: Isolate only the sharp, top-surface layer of the sponge
    if top_layer_focus:
        # Calculate local texture sharpness via a morphological range filter
        kernel_focus = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        local_max = cv2.dilate(image, kernel_focus)
        local_min = cv2.erode(image, kernel_focus)
        local_variance = cv2.absdiff(local_max, local_min)
        
        # Binary mask holding ONLY the sharp top-surface struts
        _, sharp_surface_mask = cv2.threshold(local_variance, focus_sensitivity, 255, cv2.THRESH_BINARY)
        
        # Smooth the surface mask to fill internal strut gaps
        kernel_clean = np.ones((5, 5), np.uint8)
        sharp_surface_mask = cv2.morphologyEx(sharp_surface_mask, cv2.MORPH_CLOSE, kernel_clean)
    else:
        sharp_surface_mask = np.ones_like(image) * 255

    # 2. Extract the general Strut Phase
    strut_flags = cv2.THRESH_BINARY_INV if strut_invert else cv2.THRESH_BINARY
    _, strut_mask = cv2.threshold(image, strut_threshold, 255, strut_flags)
    
    # Restrict the strut mask strictly to the sharp top layer
    top_struts = cv2.bitwise_and(strut_mask, sharp_surface_mask)
    
    # 3. Clean and isolate the core body of the top struts
    if pore_mask is not None:
        strut_body = cv2.bitwise_and(top_struts, cv2.bitwise_not(pore_mask))
    else:
        strut_body = top_struts.copy()
        
    # Erode edges to ensure boundary lines aren't caught as cracks
    kernel_erode = np.ones((3, 3), np.uint8)
    safe_strut_zone = cv2.erode(strut_body, kernel_erode, iterations=1)

    # 4. Extract Line Structures using Hessian Matrices
    hxx, hxy, hyy = hessian_matrix(image, sigma=crack_sigma)
    i1, i2 = hessian_matrix_eigvals([hxx, hxy, hyy])
    
    if crack_polarity == "Dark Cracks":
        raw_intensity = i1  
    else:
        raw_intensity = -i2 

    hessian_cutoff = threshold / 50.0
    ridge_map = np.where(raw_intensity > hessian_cutoff, 255, 0).astype(np.uint8)

    # 5. Mask out all ridges that fall outside the top surface zone
    valid_cracks = cv2.bitwise_and(ridge_map, safe_strut_zone)

    # 6. Geometrical Filter
    labeled_cracks = measure.label(valid_cracks > 0)
    props = measure.regionprops(labeled_cracks)
    
    final_crack_mask = np.zeros_like(valid_cracks)
    for prop in props:
        if prop.area < min_crack_size: 
            continue
            
        if prop.eccentricity >= min_eccentricity:
            for coord in prop.coords:
                final_crack_mask[coord, coord] = 255

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

    mean_length = np.mean(lengths) if len(lengths) > 0 else 0
    max_length = np.max(lengths) if len(lengths) > 0 else 0
    mean_ar = np.mean(aspect_ratios) if len(aspect_ratios) > 0 else 0
    density = len(lengths) / total_area if total_area > 0 else 0

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
    if not np.any(binary):
        return np.zeros_like(mask, dtype=float), 0.0
    distance = distance_transform_edt(binary)
    avg_thickness = np.mean(distance[binary]) * 2
    return distance, avg_thickness

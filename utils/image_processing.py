import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.morphology import skeletonize

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
    """
    Analyzes pores and filters out broken, half-opened, or jagged shapes.
    Only counts closed, round/oval pores using circularity constraints.
    """
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    validated_areas = []
    cleaned_pore_mask = np.zeros_like(mask)
    
    for prop in props:
        if prop.area < 15:
            continue
            
        # Circularity = (4 * pi * Area) / (Perimeter^2)
        # Perfect circle = 1.0. Broken/jagged shapes approach 0.0.
        perimeter = prop.perimeter
        if perimeter == 0:
            continue
            
        circularity = (4 * np.pi * prop.area) / (perimeter ** 2)
        
        # Keep only round/oval closed pores (reject broken borders)
        if circularity >= 0.55:  # Relaxed slightly to catch natural ovals
            validated_areas.append(prop.area)
            for coord in prop.coords:
                cleaned_pore_mask[coord[0], coord[1]] = 255

    # Update the mask array contents in-place for subsequent processes
    mask[:] = cleaned_pore_mask

    mean_area = np.mean(validated_areas) if len(validated_areas) > 0 else 0
    return {
        "areas": validated_areas,
        "mean_area": mean_area,
        "count": len(validated_areas)
    }

def detect_cracks(
    image, 
    threshold=80, 
    pore_mask=None, 
    strut_threshold=40,
    strut_invert=False,
    crack_polarity="Dark Cracks",
    crack_sigma=3.0,
    min_eccentricity=0.70,
    min_crack_size=5
):
    """
    Identifies cracks based on proximity and connectivity to validated pores.
    Traces structural linear valleys propagating through struts.
    """
    # 1. Isolate the Solid Struts
    strut_flags = cv2.THRESH_BINARY_INV if strut_invert else cv2.THRESH_BINARY
    _, strut_mask = cv2.threshold(image, strut_threshold, 255, strut_flags)
    
    if pore_mask is not None:
        strut_body = cv2.bitwise_and(strut_mask, cv2.bitwise_not(pore_mask))
    else:
        strut_body = strut_mask.copy()

    # 2. Extract topological valley structures (Morphological profiles)
    kernel_size = int(max(3, 2 * round(crack_sigma) + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if crack_polarity == "Dark Cracks":
        topological_lines = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    else:
        topological_lines = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
    _, trace_mask = cv2.threshold(topological_lines, int(threshold * 0.3), 255, cv2.THRESH_BINARY)
    trace_mask = cv2.bitwise_and(trace_mask, strut_body)

    # 3. Validation: Verify connectivity to validated round pores
    if pore_mask is not None and np.any(pore_mask > 0):
        # Dilate validated pores to create an intersection boundary zone
        pore_zones = cv2.dilate(pore_mask, np.ones((5, 5), np.uint8), iterations=1)
        
        labeled_lines = measure.label(trace_mask > 0)
        props = measure.regionprops(labeled_lines)
        
        final_crack_mask = np.zeros_like(trace_mask)
        
        for prop in props:
            if prop.area < min_crack_size:
                continue
                
            # Isolate the individual line component
            single_line_mask = np.zeros_like(trace_mask)
            for coord in prop.coords:
                single_line_mask[coord[0], coord[1]] = 255
                
            # Assess if this line intersects a pore zone boundary
            touches_pore = cv2.bitwise_and(single_line_mask, pore_zones)
            
            # CRACK CRITERIA: High linear profile OR anchored to an active pore node
            if prop.eccentricity >= min_eccentricity or np.any(touches_pore > 0):
                final_crack_mask = cv2.bitwise_or(final_crack_mask, single_line_mask)
        return final_crack_mask

    return trace_mask

def crack_analysis(mask):
    binary = mask > 0
    if not np.any(binary):
        return {"lengths": [], "mean_length": 0, "max_length": 0, "mean_aspect_ratio": 0, "count": 0, "density": 0}
        
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

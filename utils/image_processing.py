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
    Analyzes pores and filters out broken, unclosed, or jagged shapes.
    Only counts round or cleanly oval ones using circularity constraints.
    """
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    validated_areas = []
    cleaned_pore_mask = np.zeros_like(mask)
    
    for prop in props:
        if prop.area < 15:
            continue
            
        # Calculate Circularity Metric = (4 * pi * Area) / (Perimeter^2)
        # Perfect circle = 1.0, Jagged or broken shapes approach 0.0
        perimeter = prop.perimeter
        if perimeter == 0:
            continue
            
        circularity = (4 * np.pi * prop.area) / (perimeter ** 2)
        
        # Keep only round/oval closed pores (reject broken/half-opened boundaries)
        if circularity >= 0.65:
            validated_areas.append(prop.area)
            for coord in prop.coords:
                cleaned_pore_mask[coord[0], coord[1]] = 255

    # Dynamically update the mask array content in-place to affect subsequent steps
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
    crack_sigma=1.0,
    min_eccentricity=0.75,
    min_crack_size=3
):
    """
    Finds cracks based on topological connectivity instead of just pixel brightness.
    Traces paths propagating through struts starting from validated pores.
    """
    # 1. Isolate the Solid Struts
    strut_flags = cv2.THRESH_BINARY_INV if strut_invert else cv2.THRESH_BINARY
    _, strut_mask = cv2.threshold(image, strut_threshold, 255, strut_flags)
    
    # Clean up the strut mask
    if pore_mask is not None:
        strut_body = cv2.bitwise_and(strut_mask, cv2.bitwise_not(pore_mask))
    else:
        strut_body = strut_mask.copy()

    # 2. Adaptive Local Valley Tracing (Dynamic Range Profiling)
    # This acts as an edge-independent profile tracer
    kernel_size = int(max(3, 2 * round(crack_sigma) + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if crack_polarity == "Dark Cracks":
        # Black hat extraction captures narrow lines darker than surroundings
        topological_lines = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    else:
        # Top hat extraction captures narrow lines brighter than surroundings
        topological_lines = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
    # Apply a local adaptive threshold over the structural valley tracing
    _, trace_mask = cv2.threshold(topological_lines, int(threshold * 0.3), 255, cv2.THRESH_BINARY)
    trace_mask = cv2.bitwise_and(trace_mask, strut_body)

    # 3. Connectivity Verification Logic: Connect to Validated Pores
    if pore_mask is not None and np.any(pore_mask > 0):
        # Dilate the round pores slightly to create an intersection hit-zone
        pore_zones = cv2.dilate(pore_mask, np.ones((5, 5), np.uint8), iterations=1)
        
        # Label every line segment found by our trace mask
        labeled_lines = measure.label(trace_mask > 0)
        props = measure.regionprops(labeled_lines)
        
        final_crack_mask = np.zeros_like(trace_mask)
        
        for prop in props:
            if prop.area < min_crack_size:
                continue
                
            # Create a single-element temporary validation mask
            single_line_mask = np.zeros_like(trace_mask)
            for coord in prop.coords:
                single_line_mask[coord[0], coord[1]] = 255
                
            # Check if this specific structural line physically touches a pore boundary hit-zone
            touches_pore = cv2.bitwise_and(single_line_mask, pore_zones)
            
            # CRACK CONDITION: Line must have linear elongation OR connect directly to a round pore
            if prop.eccentricity >= min_eccentricity or np.any(touches_pore > 0):
                final_crack_mask = cv2.bitwise_or(final_crack_mask, single_line_mask)
        return final_crack_mask

    # Fallback configuration if no pores exist
    return trace_mask

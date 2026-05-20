import streamlit as st
import numpy as np
import cv2
from streamlit_cropper import st_cropper
from PIL import Image
import queue

# Page configuration
st.set_page_config(page_title="Pathfinding Crack Tracker", layout="centered")

st.title("🧽 Pathfinding Flow-Field Crack Tracker")
st.write("Tracks fractures by treating the strut as a structural wall and searching for low-resistance path shortcuts cutting across it.")

# Sidebar Settings
st.sidebar.header("Pathfinding Mechanics")

wall_density = st.sidebar.slider(
    "Strut Solid Resistance", 
    min_value=10, 
    max_value=250, 
    value=180,
    help="Determines how hard the solid material resists path crossing. Higher values focus tracking inside tight cuts."
)

scan_spacing = st.sidebar.slider(
    "Scan Step Density (Pixels)", 
    min_value=5, 
    max_value=50, 
    value=20,
    help="Controls how many entry/exit checkpoint paths are cross-checked over the material surface."
)

uploaded_file = st.file_uploader("Upload sponge surface image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    
    st.subheader("1. Crop the Active Fracture Zone")
    cropped_pil = st_cropper(pil_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    orig_img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    
    st.subheader("2. Tracked Failure Flow-Path")
    
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    # Apply a heavy bilateral filter to erase the texture patterns of lower layers while keeping the foreground strut edges crisp
    smoothed = cv2.bilateralFilter(gray, 15, 80, 80)
    
    # Generate a Cost Map: Brighter strut pixels have high cost. Dark crack pixels have low cost.
    # We invert the grayscale so that the background/strut becomes a high-cost mountain ridge
    cost_map = np.uint8(np.clip(smoothed.astype(np.int32) * (wall_density / 255.0), 1, 255))
    
    H, W = cost_map.shape
    crack_overlay = orig_img.copy()
    shortcut_map = np.zeros_like(gray)
    
    # Dijkstra/A* Pathfinding implementation tailored to find the path of least resistance across the columns
    def find_leak_path(cost_grid, start_y, end_y):
        # priority queue stores: (total_cost, current_x, current_y, path_history)
        pq = queue.PriorityQueue()
        pq.put((int(cost_grid[start_y, 0]), 0, start_y, [(0, start_y)]))
        
        visited = np.zeros_like(cost_grid, dtype=bool)
        visited[start_y, 0] = True
        
        while not pq.empty():
            current_cost, x, y, path = pq.get()
            
            # Reach the opposite side of the strut window
            if x == W - 1:
                return path, current_cost
                
            # Check 8-way directional movement neighbors
            for dx, dy in [(1,0), (1,1), (1,-1), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if not visited[ny, nx]:
                        visited[ny, nx] = True
                        step_cost = int(cost_grid[ny, nx])
                        # If a diagnostic path cuts diagonally, give it a tiny buffer favor
                        if dx != 0 and dy != 0:
                            step_cost = int(step_cost * 1.4)
                        pq.put((current_cost + step_cost, nx, ny, path + [(nx, ny)]))
        return None, float('inf')

    # Step through the image rows horizontally to spot where paths naturally converge into the crack channel
    all_paths = []
    costs = []
    
    for y_start in range(10, H - 10, scan_spacing):
        path, final_cost = find_leak_path(cost_map, y_start, y_start)
        if path:
            all_paths.append(path)
            costs.append(final_cost)
            
    if costs:
        # The true crack path represents the global minimum path resistance across the entire sample window
        min_cost = min(costs)
        avg_cost = np.mean(costs)
        
        # A structural breach creates a massive drop in traversal cost compared to healthy sections
        for path, path_cost in zip(all_paths, costs):
            # If this path configuration bypasses the 'mountain' with lower relative resistance, draw it
            if path_cost < avg_cost * 0.75:
                for idx in range(len(path) - 1):
                    cv2.line(crack_overlay, path[idx], path[idx+1], (0, 0, 255), 2)
                    cv2.line(shortcut_map, path[idx], path[idx+1], 255, 2)
                    
    # Display Output Results
    st.image(crack_overlay, channels="BGR", caption="Least Resistance Flow-Line (Forced Crack Path Tracking)", use_container_width=True)
    
    with st.expander("Show Resistance Cost Map Diagnostics"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(cost_map, caption="Terrain Cost Map (Dark Valleys = Easy Paths)", use_container_width=True)
        with col2:
            st.image(shortcut_map, caption="Isolated Path Convergence Vector", use_container_width=True)
else:
    st.info("Upload your multi-layered sponge sample view to track the structural shortcut paths.")

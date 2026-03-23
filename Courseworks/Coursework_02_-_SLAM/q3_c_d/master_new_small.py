import json
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.neighbors import NearestNeighbors

from graph import Graph
from pose_se2 import PoseSE2
from loop_closure import LoopClosureDetector
import icp

import pygame


# GITHUB ICP PATCH (For Loop Closure Verification)
#The main reference for the ICP implementation is the HobbySingh repository, 
# The following code defines the custom ICP function and replaces the original one in the imported module.
def custom_icp(A, B, init_pose=None, max_iterations=40, tolerance=0.001):
    m = A.shape[1]
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    iters = 0
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst[:m, :].T)
    
    for i in range(max_iterations):
        iters = i
        distances, indices = neigh.kneighbors(src[:m, :].T, return_distance=True)
        distances = distances.ravel()
        indices = indices.ravel()
        valid = (distances < 1.0) & (np.linalg.norm(src[:m, :], axis=0) < 80)
        if np.sum(valid) < 3: break
        
        filtered_src = src[:, valid]
        filtered_indices = indices[valid]
        T, _, _ = icp.best_fit_transform(filtered_src[:m, :].T, dst[:m, filtered_indices].T)
        src = np.dot(T, src)
        
        mean_error = np.mean(distances[valid])
        if np.abs(prev_error - mean_error) < tolerance: break
        prev_error = mean_error

    T, _, _ = icp.best_fit_transform(A, src[:m, :].T)
    return T, distances, iters, np.eye(3)

icp.icp = custom_icp


# LAB 5 POINT-TO-PLANE ICP (For Odometry)
ICP_MAX_ITER = 10
CORRESPONDENCE_THRESH = 0.5
KEYFRAME_DIST_THRESH = 0.2
KEYFRAME_ANGLE_THRESH = 0.2
LOCAL_MAP_SIZE = 20

def estimate_normals_pca(points, k=5):
    if len(points) < k + 1:
        return np.zeros((len(points), 2))
    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(points)
    _, indices_all = neigh.kneighbors(points)
    normals = np.zeros((points.shape[0], 2))
    for i in range(points.shape[0]):
        neighbor_points = points[indices_all[i]]
        centered = neighbor_points - np.mean(neighbor_points, axis=0)
        cov = np.dot(centered.T, centered) / k
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, 0]
        if np.dot(normal, points[i]) < 0:
            normal = -normal
        normals[i] = normal
    return normals

def solve_point_to_plane(src, dst, dst_normals):
    A = []
    b = []
    for i in range(len(src)):
        s = src[i]
        d = dst[i]
        n = dst_normals[i]
        cross_term = s[0] * n[1] - s[1] * n[0]
        A.append([cross_term, n[0], n[1]])
        b.append(np.dot(d - s, n))
    if not A: return np.identity(3)
    A = np.array(A)
    b = np.array(b)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    c, s = np.cos(x[0]), np.sin(x[0])
    R = np.array([[c, -s], [s, c]])
    T = np.identity(3)
    T[:2, :2] = R
    T[:2, 2] = [x[1], x[2]]
    return T

def icp_scan_to_map(src_points, map_points, map_normals, init_pose_guess):
    m = src_points.shape[1]
    src_h = np.ones((m+1, src_points.shape[0])) 
    src_h[:m,:] = np.copy(src_points.T)
    current_global_pose = np.copy(init_pose_guess)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(map_points)
    
    for i in range(ICP_MAX_ITER):
        src_global_h = np.dot(current_global_pose, src_h)
        src_global = src_global_h[:2, :].T
        distances, indices = neigh.kneighbors(src_global, return_distance=True)
        distances = distances.ravel()
        indices = indices.ravel()
        
        mask = distances < CORRESPONDENCE_THRESH
        if np.sum(mask) < 10: break
        
        src_valid = src_global[mask]
        dst_valid = map_points[indices[mask]]
        normals_valid = map_normals[indices[mask]]
        
        T_delta = solve_point_to_plane(src_valid, dst_valid, normals_valid)
        current_global_pose = np.dot(T_delta, current_global_pose)
        
        if np.linalg.norm(T_delta[:2, 2]) < 0.001 and abs(np.arctan2(T_delta[1,0], T_delta[0,0])) < 0.001:
            break
    return current_global_pose



# Configuration
# Uncomment the dataset you wish to run:

# 1. Small Chair Room
#JSON_FILE = r'C:\dev\COMP0249_coursework_1\Courseworks\Coursework_02_-_SLAM\small_chair_room\small_chair_room\charvi_small_chair_room.json' 
#OUTPUT_NAME = "small_chair_room"
#SEARCH_RADIUS = 2.0

# 2. Indoor Big Loop
JSON_FILE = r'C:\dev\COMP0249_coursework_1\Courseworks\Coursework_02_-_SLAM\indoor_big_loop\indoor_big_loop\indoor_big_loop.json' 
OUTPUT_NAME = "indoor_big_loop"
SEARCH_RADIUS = 10.0

# 3. Outdoor Area
#JSON_FILE = r'C:\dev\COMP0249_coursework_1\Courseworks\Coursework_02_-_SLAM\outdoor_area\outdoor_area\charvi_outside.json' 
#OUTPUT_NAME = "outdoor_area"
#SEARCH_RADIUS = 10.0

MIN_RANGE_MM = 200.0
MAX_RANGE_MM = 12000.0
VOXEL_SIZE = 0.05 
SCAN_SKIP = 1

BLIND_SPOT_MIN = 135.0
BLIND_SPOT_MAX = 225.0

LIVE_VISUALIZATION = True
GENERATE_OCCUPANCY_GRID = True


# Helper Functions
def load_scans_from_json(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return []
    with open(filename, "r") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "scans" in data:
            return data["scans"]
        elif isinstance(data, list):
            return data
    except: pass
    scans = []
    for line in content.splitlines():
        if line.strip(): scans.append(json.loads(line.strip()))
    return scans

def voxel_downsample(points, voxel_size):
    if len(points) == 0: return points
    voxel_idx = np.floor(points / voxel_size).astype(int)
    unique_voxels, indices = np.unique(voxel_idx, axis=0, return_index=True)
    return points[indices]

def process_scan(scan_data):
    raw = np.array(scan_data)
    if len(raw) == 0: return None
    distances = raw[:, 2]
    angles = raw[:, 1]
    
    dist_mask = (distances > MIN_RANGE_MM) & (distances < MAX_RANGE_MM)
    angle_mask = (angles < BLIND_SPOT_MIN) | (angles > BLIND_SPOT_MAX) 
    mask = dist_mask & angle_mask
    
    if np.sum(mask) < 10: return None
    
    angles_rad = np.radians(raw[mask, 1])
    dists_m = raw[mask, 2] / 1000.0 
    
    x = dists_m * np.cos(angles_rad)
    y = dists_m * np.sin(angles_rad)
    points = np.column_stack((x, y))
    
    return voxel_downsample(points, VOXEL_SIZE)

def build_occupancy_grid(pose_graph, all_lasers, output_name):
    print("\GENERATING OCCUPANCY GRID")
    pygame.init()
    
    CELL_SIZE_M = 0.02
    num_poses = len(all_lasers)
    
    # 1. Dynamically calculate the map bounds based on the trajectory!
    all_px = []
    all_py = []
    for i in range(num_poses):
        try:
            pose = pose_graph.get_pose(i).arr.flatten()
            all_px.append(pose[0])
            all_py.append(pose[1])
        except Exception:
            break
            
    min_x, max_x = min(all_px), max(all_px)
    min_y, max_y = min(all_py), max(all_py)
    
    # Add an 8-meter padding on all sides for better fitting
    padding_m = 8.0
    min_x -= padding_m
    max_x += padding_m
    min_y -= padding_m
    max_y += padding_m
    
    width_px = int((max_x - min_x) / CELL_SIZE_M)
    height_px = int((max_y - min_y) / CELL_SIZE_M)
    MAP_SIZE = (width_px, height_px)

    # Gray background (Unknown)
    map_surface = pygame.Surface(MAP_SIZE)
    map_surface.fill((128, 128, 128))

    CONFIDENCE_FREE = (15, 15, 15)      # 'Eraser' to white
    CONFIDENCE_OCCUPIED = (50, 50, 50)  # 'Pen' to black

    print(f"Ray-casting trajectory (Dynamic Map Size: {width_px}x{height_px} px)...")
    
    trajectory_points = [] 

    for i in tqdm(range(len(all_px))):
        pose = pose_graph.get_pose(i).arr.flatten()
        px, py, pyaw = pose[0], pose[1], pose[2]
        
        # Map physical coordinates to pixel coordinates using our dynamic bounds
        rob_px = int((px - min_x) / CELL_SIZE_M)
        rob_py = int((py - min_y) / CELL_SIZE_M)
        
        # Save the pixel position for drawing later
        trajectory_points.append((rob_px, rob_py))

        flash_surface = pygame.Surface(MAP_SIZE)
        flash_surface.fill((0, 0, 0))
        hits_surface = pygame.Surface(MAP_SIZE)
        hits_surface.fill((0, 0, 0))

        scan = all_lasers[i]
        cos_th = np.cos(pyaw)
        sin_th = np.sin(pyaw)

        for pt in scan:
            gx = (pt[0] * cos_th - pt[1] * sin_th) + px
            gy = (pt[0] * sin_th + pt[1] * cos_th) + py

            
            hit_px = int((gx - min_x) / CELL_SIZE_M)
            hit_py = int((gy - min_y) / CELL_SIZE_M)

            # Only draw if within bounds
            if 0 <= hit_px < MAP_SIZE[0] and 0 <= hit_py < MAP_SIZE[1]:
                # Ray-cast free space
                pygame.draw.line(flash_surface, CONFIDENCE_FREE, (rob_px, rob_py), (hit_px, hit_py), 3)
                # Draw occupied obstacle
                pygame.draw.circle(hits_surface, CONFIDENCE_OCCUPIED, (hit_px, hit_py), 4)

        # Blend the layers using Lab 08 logic
        map_surface.blit(flash_surface, (0, 0), special_flags=pygame.BLEND_ADD)
        map_surface.blit(hits_surface, (0, 0), special_flags=pygame.BLEND_SUB)

        
        if i % 20 == 0:
            pygame.event.pump()
            pygame.time.delay(1)

    # Draw the trajectory on top of the finished map!
    if len(trajectory_points) > 1:
        # Draw red trajectory line
        pygame.draw.lines(map_surface, (255, 0, 0), False, trajectory_points, 3)
        # Draw green dot at start
        pygame.draw.circle(map_surface, (0, 255, 0), trajectory_points[0], 6)
        # Draw blue dot at end
        pygame.draw.circle(map_surface, (0, 0, 255), trajectory_points[-1], 6)

    # Save to file
    filename = f"{output_name}_Occupancy_Grid.png"
    pygame.image.save(map_surface, filename)
    print(f"Map saved as {filename}")
    pygame.quit()
    
    return filename

def draw_live_graph(pose_graph, all_lasers, max_idx, ax, title):
    """ Extracts and plots the current state of the Pose Graph AND Map in real-time. """
    ax.clear()
    xs, ys = [], []
    map_x, map_y = [], []
    
    for i in range(max_idx + 1):
        try:
            pose = pose_graph.get_pose(i).arr.flatten()
            px, py, pyaw = pose[0], pose[1], pose[2]
            xs.append(px)
            ys.append(py)
            
            # Transform local laser scans into the global coordinate frame
            if i < len(all_lasers):
                scan = all_lasers[i]
                cos_th, sin_th = np.cos(pyaw), np.sin(pyaw)
                
                gx = scan[::2, 0] * cos_th - scan[::2, 1] * sin_th + px
                gy = scan[::2, 0] * sin_th + scan[::2, 1] * cos_th + py
                
                map_x.extend(gx)
                map_y.extend(gy)
        except:
            pass
            
    # Plot Environment Map (Blue)
    if len(map_x) > 0:
        ax.scatter(map_x, map_y, s=1, c='blue', alpha=0.15, zorder=1)
        
    # Plot Trajectory
    if len(xs) > 0:
        ax.plot(xs, ys, 'r-', linewidth=2.0, zorder=5)
        ax.plot(xs[0], ys[0], 'ro', markersize=8, label='Start', zorder=10)
        ax.plot(xs[-1], ys[-1], 'gx', markersize=8, label='End', zorder=10)
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    plt.pause(0.001)


# Main SLAM Loop
def run_robust_slam():
    print(f"\nLoading data from {JSON_FILE}...")
    raw_scans = load_scans_from_json(JSON_FILE)
    if not raw_scans: return
        
    pose_graph = Graph([], [])
    detector = LoopClosureDetector(search_radius=SEARCH_RADIUS, icp_error_thresh=0.10, temporal_skip=50)

    current_pose_mat = np.identity(3)
    last_keyframe_pose = np.identity(3)
    pose_graph.add_vertex(0, PoseSE2.from_rt_matrix(current_pose_mat))
    
    keyframe_buffer = [] 
    all_lasers = [] 
    vertex_idx = 1
    first_scan_done = False
    
    print(f"Starting Robust Point-to-Plane SLAM for {OUTPUT_NAME}...")

    # Initialize Live Plotting Window 
    if LIVE_VISUALIZATION:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in tqdm(range(0, len(raw_scans), SCAN_SKIP)):
        curr_xy = process_scan(raw_scans[i])
        if curr_xy is None: continue
            
        if not first_scan_done:
            curr_h = np.ones((3, curr_xy.shape[0]))
            curr_h[:2,:] = curr_xy.T
            curr_global = np.dot(current_pose_mat, curr_h)[:2,:].T
            
            normals = estimate_normals_pca(curr_global)
            keyframe_buffer.append((curr_global, normals))
            all_lasers.append(curr_xy)
            first_scan_done = True
            continue
            
        active_points = np.vstack([k[0] for k in keyframe_buffer])
        active_normals = np.vstack([k[1] for k in keyframe_buffer])
        
        # LAB 5 POINT-TO-PLANE ICP!
        current_pose_mat = icp_scan_to_map(curr_xy, active_points, active_normals, current_pose_mat)
        
        # KEYFRAME CHECK
        delta_T = np.dot(np.linalg.inv(last_keyframe_pose), current_pose_mat)
        dx, dy = delta_T[0,2], delta_T[1,2]
        dtheta = np.arctan2(delta_T[1,0], delta_T[0,0])
        dist_moved = np.sqrt(dx**2 + dy**2)
        
        if dist_moved > KEYFRAME_DIST_THRESH or abs(dtheta) > KEYFRAME_ANGLE_THRESH:
            pose_graph.add_vertex(vertex_idx, PoseSE2.from_rt_matrix(current_pose_mat))
            pose_graph.add_edge([vertex_idx - 1, vertex_idx], PoseSE2.from_rt_matrix(delta_T), np.eye(3))
            all_lasers.append(curr_xy)
            
            # Live Plot Update
            if LIVE_VISUALIZATION and vertex_idx % 5 == 0:
                draw_live_graph(pose_graph, all_lasers, vertex_idx, ax, f"Live SLAM Tracking... ({OUTPUT_NAME})")
            

           # if vertex_idx % 5 == 0: #uncomment for 'before' graph
                # TEMPORARILY DISABLED FOR 'BEFORE' GRAPH
              #  loop_found = False
            # Check for Loop Closure on Keyframes
            if vertex_idx % 5 == 0:
                loop_found = detector.find_loop_closure(
                   curr_pose=PoseSE2.from_rt_matrix(current_pose_mat),
                    curr_idx=vertex_idx,
                    all_lasers=all_lasers,
                    pose_graph=pose_graph
                )
                
                if loop_found:
                    # Pre-snap visualization
                    if LIVE_VISUALIZATION:
                        draw_live_graph(pose_graph, all_lasers, vertex_idx, ax, "Loop Closure Detected! (Pre-Optimization)")
                        plt.pause(0.5) 
                        
                    print(f"\n[Optimizer] Loop closure! Optimizing graph at vertex {vertex_idx}...")
                    pose_graph.optimize(tol=1e-4, max_iter=20)
                    
                    opt_pose = pose_graph.get_pose(vertex_idx).arr.flatten()
                    px, py, pyaw = opt_pose[0], opt_pose[1], opt_pose[2]
                    current_pose_mat = np.array([
                        [np.cos(pyaw), -np.sin(pyaw), px],
                        [np.sin(pyaw),  np.cos(pyaw), py],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float64)
                    
                    # Post-snap visualization
                    if LIVE_VISUALIZATION:
                        draw_live_graph(pose_graph, all_lasers, vertex_idx, ax, "Graph Optimized! (Snapped)")
                        plt.pause(1.0) 
                    
                    # Clear local map so it doesn't drag the robot back into the drift!
                    keyframe_buffer = [] 

            curr_h = np.ones((3, curr_xy.shape[0]))
            curr_h[:2,:] = curr_xy.T
            curr_global = np.dot(current_pose_mat, curr_h)[:2,:].T
            curr_normals = estimate_normals_pca(curr_global)
            
            keyframe_buffer.append((curr_global, curr_normals))
            if len(keyframe_buffer) > LOCAL_MAP_SIZE:
                keyframe_buffer.pop(0)
                
            last_keyframe_pose = np.copy(current_pose_mat)
            vertex_idx += 1

    print("Finished processing scans. Running final global optimization...")
    pose_graph.optimize(tol=1e-4, max_iter=40)
    
    if LIVE_VISUALIZATION:
        draw_live_graph(pose_graph, all_lasers, vertex_idx, ax, "Final Global Optimization")
        plt.ioff() # Keep window open at the end
    
    print(f"Plotting and saving {OUTPUT_NAME}.png...")
    pose_graph.plot(title=OUTPUT_NAME)
    
    # QuestionD: Error Quantification
    print("\nQuestion D: ERROR METRICS")
    
    # 1. The Rubric Metric: Start-to-End Euclidean Distance
    start_pose = pose_graph.get_pose(0).arr.flatten()
    end_pose = pose_graph.get_pose(vertex_idx - 1).arr.flatten()
    start_to_end_dist = math.sqrt((end_pose[0] - start_pose[0])**2 + (end_pose[1] - start_pose[1])**2)
    print(f"Start-to-End Euclidean Distance: {start_to_end_dist:.4f} meters")

    # 2. Second Metric: Cross-Loop Average Separation
    # We extract all X,Y coordinates from the graph
    all_xy = np.array([pose_graph.get_pose(i).arr.flatten()[:2] for i in range(vertex_idx)])
    
    # we split the trajectory in half
    half_idx = len(all_xy) // 2
    loop1_poses = all_xy[:half_idx]
    loop2_poses = all_xy[half_idx:]
    
    # We use a KD-Tree to find the closest point in Loop 1 for every point in Loop 2
    import scipy.spatial
    tree = scipy.spatial.cKDTree(loop1_poses)
    distances, _ = tree.query(loop2_poses)
    
    mean_loop_separation = np.mean(distances)
    print(f"Mean Cross-Loop Separation:      {mean_loop_separation:.4f} meters")

    if GENERATE_OCCUPANCY_GRID:
        build_occupancy_grid(pose_graph, all_lasers, OUTPUT_NAME)
    plt.show()

if __name__ == "__main__":
    run_robust_slam()
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

# ==========================================
# CONFIG
# ==========================================
JSON_FILE = "outdoor_area/charvi_outside.json"

ICP_MAX_ITER = 10
CORRESPONDENCE_THRESH = 0.5
KEYFRAME_DIST_THRESH = 0.2
KEYFRAME_ANGLE_THRESH = 0.2
LOCAL_MAP_SIZE = 20

SENSOR_MAX_RANGE_MM = 12000.0
MIN_RANGE_MM = 200.0

BLIND_SPOT_MIN = 135.0
BLIND_SPOT_MAX = 225.0


# ==========================================
# HELPERS
# ==========================================
def load_scans_from_json(filename):
    scans = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scans.append(json.loads(line))
    return scans


def estimate_normals_pca(points, k=5):
    if len(points) < k + 1:
        return np.zeros((len(points), 2))

    neigh = NearestNeighbors(n_neighbors=k + 1)
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

    if not A:
        return np.identity(3)

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
    src_h = np.ones((m + 1, src_points.shape[0]))
    src_h[:m, :] = np.copy(src_points.T)

    current_global_pose = np.copy(init_pose_guess)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(map_points)

    for _ in range(ICP_MAX_ITER):
        src_global_h = np.dot(current_global_pose, src_h)
        src_global = src_global_h[:2, :].T

        distances, indices = neigh.kneighbors(src_global, return_distance=True)
        distances = distances.ravel()
        indices = indices.ravel()

        mask = distances < CORRESPONDENCE_THRESH
        if np.sum(mask) < 10:
            break

        src_valid = src_global[mask]
        dst_valid = map_points[indices[mask]]
        normals_valid = map_normals[indices[mask]]

        T_delta = solve_point_to_plane(src_valid, dst_valid, normals_valid)
        current_global_pose = np.dot(T_delta, current_global_pose)

        delta_trans = np.linalg.norm(T_delta[:2, 2])
        delta_rot = abs(np.arctan2(T_delta[1, 0], T_delta[0, 0]))
        if delta_trans < 0.001 and delta_rot < 0.001:
            break

    return current_global_pose


def process_scan(scan_data, max_range_mm, beam_step=1, use_blind_spot_filter=True):
    raw = np.array(scan_data)
    if len(raw) == 0:
        return None

    distances = raw[:, 2]
    angles = raw[:, 1]

    dist_mask = (distances > MIN_RANGE_MM) & (distances < max_range_mm)

    if use_blind_spot_filter:
        angle_mask = (angles < BLIND_SPOT_MIN) | (angles > BLIND_SPOT_MAX)
        mask = dist_mask & angle_mask
    else:
        mask = dist_mask

    filtered = raw[mask]
    if len(filtered) < 10:
        return None

    filtered = filtered[::beam_step]
    if len(filtered) < 10:
        return None

    angles_rad = np.radians(filtered[:, 1])
    dists_m = filtered[:, 2] / 1000.0

    x = dists_m * np.cos(angles_rad)
    y = dists_m * np.sin(angles_rad)
    points = np.column_stack((x, y))

    if len(points) < 10:
        return None

    return points


# ==========================================
# OFFLINE SLAM
# ==========================================
def run_offline_slam(scans, max_range_mm, beam_step=1, scan_skip=1, title="experiment"):
    current_pose = np.identity(3)
    last_keyframe_pose = np.identity(3)

    keyframe_buffer = []
    global_map_points = []
    trajectory = [[0.0, 0.0]]
    first_scan_done = False

    for idx, scan in enumerate(scans):
        if idx % scan_skip != 0:
            continue

        current_scan_xy = process_scan(scan, max_range_mm=max_range_mm, beam_step=beam_step)
        if current_scan_xy is None:
            continue

        if not first_scan_done:
            normals = estimate_normals_pca(current_scan_xy)
            keyframe_buffer.append((current_scan_xy, normals))
            global_map_points.append(current_scan_xy)
            first_scan_done = True
            continue

        active_points = np.vstack([k[0] for k in keyframe_buffer])
        active_normals = np.vstack([k[1] for k in keyframe_buffer])

        new_pose = icp_scan_to_map(current_scan_xy, active_points, active_normals, current_pose)
        current_pose = new_pose

        cx, cy = current_pose[0, 2], current_pose[1, 2]
        trajectory.append([cx, cy])

        delta_T = np.dot(np.linalg.inv(last_keyframe_pose), current_pose)
        dx, dy = delta_T[0, 2], delta_T[1, 2]
        dtheta = np.arctan2(delta_T[1, 0], delta_T[0, 0])
        dist_moved = np.sqrt(dx**2 + dy**2)

        if dist_moved > KEYFRAME_DIST_THRESH or abs(dtheta) > KEYFRAME_ANGLE_THRESH:
            curr_h = np.ones((3, current_scan_xy.shape[0]))
            curr_h[:2, :] = current_scan_xy.T
            curr_global = np.dot(current_pose, curr_h)[:2, :].T

            curr_normals = estimate_normals_pca(curr_global)
            keyframe_buffer.append((curr_global, curr_normals))
            global_map_points.append(curr_global)
            last_keyframe_pose = np.copy(current_pose)

            if len(keyframe_buffer) > LOCAL_MAP_SIZE:
                keyframe_buffer.pop(0)

    if len(global_map_points) == 0:
        return None, None

    all_map_pts = np.vstack(global_map_points)
    trajectory = np.array(trajectory)
    return all_map_pts, trajectory


def plot_result(map_points, trajectory, title, save_path=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(map_points[:, 0], map_points[:, 1], s=1, label="Map points", color='blue')
    plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2, label="Trajectory", color='red')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], s=50, marker='o', label="Start", color='orange')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], s=50, marker='x', label="End", color='green')
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
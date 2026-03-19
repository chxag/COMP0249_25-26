import json
import math
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
JSON_FILE = "outdoor_area/charvi_outside.json"

# Sensor limits
SENSOR_MIN_RANGE_MM = 200.0
SENSOR_MAX_RANGE_MM = 12000.0

# Map settings
MAP_DIM = 800
CELL_SIZE_MM = 20
WINDOW_MM = MAP_DIM * CELL_SIZE_MM  # 16 m x 16 m

# Blind spot
CUT_ANGLE_MIN = 135.0
CUT_ANGLE_MAX = 225.0

# Occupancy update parameters
OCCUPIED_UPDATE = 0.10
FREE_UPDATE = 0.02

# Pose search parameters
POSE_ITERS = 10


# ==========================================
# JSON LOADER
# ==========================================
def load_scans_from_json(filename):
    with open(filename, "r") as f:
        content = f.read().strip()

    # First try normal JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "scans" in data:
            return data["scans"]
        elif isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL / one scan per line
    scans = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            scans.append(json.loads(line))
    return scans


# ==========================================
# POSE ESTIMATOR
# ==========================================
class PoseEstimator:
    def __init__(self, map_dim, cell_size_mm):
        self.map_w = map_dim
        self.map_h = map_dim
        self.cell_size = cell_size_mm
        self.reset()

    def reset(self):
        self.x = (self.map_w * self.cell_size) / 2
        self.y = (self.map_h * self.cell_size) / 2
        self.theta = 0.0

    def get_pose(self):
        return self.x, self.y, self.theta

    def optimize_pose(self, scan_points, grid_map, iterations=10):
        if len(scan_points) == 0:
            return

        scan_arr = np.array(scan_points)
        angles = scan_arr[:, 0]
        dists = scan_arr[:, 1]

        local_x = dists * np.cos(angles)
        local_y = dists * np.sin(angles)

        step_xy = 30
        step_th = np.radians(0.5)

        for _ in range(iterations):
            best_score = -float("inf")
            best_pose = (self.x, self.y, self.theta)
            found_better = False

            search_range_xy = [-step_xy, 0, step_xy]
            search_range_th = [-step_th, 0, step_th]

            for dth in search_range_th:
                test_th = self.theta + dth
                cos_th = np.cos(test_th)
                sin_th = np.sin(test_th)

                for dx in search_range_xy:
                    for dy in search_range_xy:
                        test_x = self.x + dx
                        test_y = self.y + dy

                        global_x = (local_x * cos_th - local_y * sin_th) + test_x
                        global_y = (local_x * sin_th + local_y * cos_th) + test_y

                        grid_x = (global_x / self.cell_size).astype(int)
                        grid_y = (global_y / self.cell_size).astype(int)

                        valid_mask = (
                            (grid_x >= 0) & (grid_x < self.map_w) &
                            (grid_y >= 0) & (grid_y < self.map_h)
                        )

                        if np.sum(valid_mask) > 5:
                            valid_gx = grid_x[valid_mask]
                            valid_gy = grid_y[valid_mask]
                            score = np.sum(grid_map[valid_gx, valid_gy])

                            if score > best_score:
                                best_score = score
                                best_pose = (test_x, test_y, test_th)
                                found_better = True

            if found_better:
                self.x, self.y, self.theta = best_pose
            else:
                break


# ==========================================
# PREPROCESSING
# ==========================================
def voxel_downsample_polar(points_xy_mm, voxel_size_mm):
    if len(points_xy_mm) == 0 or voxel_size_mm is None or voxel_size_mm <= 0:
        return points_xy_mm

    voxel_idx = np.floor(points_xy_mm / voxel_size_mm).astype(int)
    buckets = {}

    for i, key in enumerate(map(tuple, voxel_idx)):
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(points_xy_mm[i])

    reduced = np.array([np.mean(v, axis=0) for v in buckets.values()])
    return reduced


def preprocess_scan(
    scan,
    max_range_mm,
    beam_step=1,
    voxel_size_mm=None,
    use_blind_spot=True,
):
    valid_points = []

    for (_, angle, distance) in scan:
        if distance < SENSOR_MIN_RANGE_MM or distance > max_range_mm:
            continue

        if use_blind_spot and (CUT_ANGLE_MIN <= angle <= CUT_ANGLE_MAX):
            continue

        valid_points.append((angle, distance))

    if len(valid_points) < 5:
        return []

    # Angular downsampling
    valid_points = valid_points[::beam_step]

    if len(valid_points) < 5:
        return []

    # Convert to XY in local frame for optional voxel filtering
    angles_rad = np.radians([p[0] for p in valid_points])
    dists = np.array([p[1] for p in valid_points])

    x = dists * np.cos(angles_rad)
    y = dists * np.sin(angles_rad)
    xy = np.column_stack((x, y))

    # Voxel/grid downsampling in Cartesian space
    if voxel_size_mm is not None:
        xy = voxel_downsample_polar(xy, voxel_size_mm)

    if len(xy) < 5:
        return []

    # Convert back to (angle_rad, distance_mm)
    processed = []
    for px, py in xy:
        angle_rad = math.atan2(py, px)
        dist = math.hypot(px, py)
        processed.append((angle_rad, dist))

    return processed


# ==========================================
# MAP UPDATE
# ==========================================
def bresenham_line(x0, y0, x1, y1):
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x1, y1))
    return points


def update_occupancy_grid(grid, robot_px, robot_py, scan_points, pose, cell_size_mm):
    curr_x, curr_y, curr_th = pose
    cos_th = math.cos(curr_th)
    sin_th = math.sin(curr_th)

    for angle_rad, dist in scan_points:
        lx = dist * math.cos(angle_rad)
        ly = dist * math.sin(angle_rad)

        gx_mm = (lx * cos_th - ly * sin_th) + curr_x
        gy_mm = (lx * sin_th + ly * cos_th) + curr_y

        px = int(gx_mm / cell_size_mm)
        py = int(gy_mm / cell_size_mm)

        if 0 <= px < grid.shape[0] and 0 <= py < grid.shape[1]:
            ray_cells = bresenham_line(robot_px, robot_py, px, py)

            # Free space update along ray except endpoint
            for cx, cy in ray_cells[:-1]:
                if 0 <= cx < grid.shape[0] and 0 <= cy < grid.shape[1]:
                    grid[cx, cy] = max(0.0, grid[cx, cy] - FREE_UPDATE)

            # Occupied update at endpoint
            grid[px, py] = min(1.0, grid[px, py] + OCCUPIED_UPDATE)


# ==========================================
# OFFLINE SLAM RUNNER
# ==========================================
def run_offline_occupancy_slam(
    scans,
    max_range_mm,
    beam_step=1,
    voxel_size_mm=None,
    scan_skip=1,
):
    occupancy_grid = np.full((MAP_DIM, MAP_DIM), 0.5, dtype=np.float32)
    trajectory_points = []

    estimator = PoseEstimator(MAP_DIM, CELL_SIZE_MM)

    for idx, scan in enumerate(scans):
        if idx % scan_skip != 0:
            continue

        processed_scan = preprocess_scan(
            scan,
            max_range_mm=max_range_mm,
            beam_step=beam_step,
            voxel_size_mm=voxel_size_mm,
            use_blind_spot=True,
        )

        if not processed_scan:
            continue

        estimator.optimize_pose(processed_scan, occupancy_grid, iterations=POSE_ITERS)
        curr_x, curr_y, curr_th = estimator.get_pose()

        rob_px = int(curr_x / CELL_SIZE_MM)
        rob_py = int(curr_y / CELL_SIZE_MM)

        if 0 <= rob_px < MAP_DIM and 0 <= rob_py < MAP_DIM:
            trajectory_points.append((rob_px, rob_py))

        update_occupancy_grid(
            occupancy_grid,
            rob_px,
            rob_py,
            processed_scan,
            (curr_x, curr_y, curr_th),
            CELL_SIZE_MM,
        )

    return occupancy_grid, np.array(trajectory_points), estimator.get_pose()


# ==========================================
# PLOTTING
# ==========================================
def plot_occupancy_result(grid, trajectory, title, save_path=None):
    plt.figure(figsize=(8, 8))

    # transpose for display orientation
    plt.imshow(
        grid.T,
        cmap="gray",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        extent=[0, MAP_DIM * CELL_SIZE_MM / 1000.0, 0, MAP_DIM * CELL_SIZE_MM / 1000.0],
    )

    if len(trajectory) > 1:
        tx = trajectory[:, 0] * CELL_SIZE_MM / 1000.0
        ty = trajectory[:, 1] * CELL_SIZE_MM / 1000.0
        plt.plot(tx, ty, linewidth=2, label="Trajectory", color="red")
        plt.scatter(tx[0], ty[0], s=50, marker="o", label="Start", color="orange")
        plt.scatter(tx[-1], ty[-1], s=50, marker="x", label="End", color="green")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ==========================================
# EXPERIMENTS
# ==========================================
if __name__ == "__main__":
    scans = load_scans_from_json(JSON_FILE)
    print(f"Loaded {len(scans)} scans")

    experiments = [
        # 1. Range
        {
            "title": "Occupancy Grid - Range 4m",
            "max_range_mm": 4000.0,
            "beam_step": 1,
            "voxel_size_mm": None,
            "scan_skip": 1,
            "save": "outdoor_area/occupancy_grid/occ_range_4m.png",
        },
        {
            "title": "Occupancy Grid - Range 12m",
            "max_range_mm": 12000.0,
            "beam_step": 1,
            "voxel_size_mm": None,
            "scan_skip": 1,
            "save": "outdoor_area/occupancy_grid/occ_range_12m.png",
        },

        # 2. Angular resolution
        {
            "title": "Occupancy Grid - Full Scan",
            "max_range_mm": 12000.0,
            "beam_step": 1,
            "voxel_size_mm": None,
            "scan_skip": 1,
            "save": "outdoor_area/occupancy_grid/occ_angular_full.png",
        },
        {
            "title": "Occupancy Grid - Every 2nd Beam",
            "max_range_mm": 12000.0,
            "beam_step": 2,
            "voxel_size_mm": None,
            "scan_skip": 1,
            "save": "outdoor_area/occupancy_grid/occ_angular_n2.png",
        },
        {
            "title": "Occupancy Grid - Every 3rd Beam",
            "max_range_mm": 12000.0,
            "beam_step": 3,
            "voxel_size_mm": None,
            "scan_skip": 1,
            "save": "outdoor_area/occupancy_grid/occ_angular_n3.png",
        },

        # 3. Voxel/grid downsampling
        {
            "title": "Occupancy Grid - Voxel 50 mm",
            "max_range_mm": 12000.0,
            "beam_step": 1,
            "voxel_size_mm": 50.0,
            "scan_skip": 1,
            "save": "outdoor_area/occupancy_grid/occ_voxel_50mm.png",
        },
        {
            "title": "Occupancy Grid - Voxel 100 mm",
            "max_range_mm": 12000.0,
            "beam_step": 1,
            "voxel_size_mm": 100.0,
            "scan_skip": 1,
            "save": "outdoor_area/occupancy_grid/occ_voxel_100mm.png",
        },

        # 4. Scan rate
        {
            "title": "Occupancy Grid - Full Rate",
            "max_range_mm": 12000.0,
            "beam_step": 1,
            "voxel_size_mm": None,
            "scan_skip": 1,
            "save": "outdoor_area/occupancy_grid/occ_scanrate_full.png",
        },
        {
            "title": "Occupancy Grid - 50% Rate",
            "max_range_mm": 12000.0,
            "beam_step": 1,
            "voxel_size_mm": None,
            "scan_skip": 2,
            "save": "outdoor_area/occupancy_grid/occ_scanrate_50.png",
        },
        {
            "title": "Occupancy Grid - 1 in 3 Scans",
            "max_range_mm": 12000.0,
            "beam_step": 1,
            "voxel_size_mm": None,
            "scan_skip": 3,
            "save": "outdoor_area/occupancy_grid/occ_scanrate_33.png",
        },
    ]

    for exp in experiments:
        print(f"Running {exp['title']}")

        grid, traj, final_pose = run_offline_occupancy_slam(
            scans,
            max_range_mm=exp["max_range_mm"],
            beam_step=exp["beam_step"],
            voxel_size_mm=exp["voxel_size_mm"],
            scan_skip=exp["scan_skip"],
        )

        plot_occupancy_result(grid, traj, exp["title"], exp["save"])
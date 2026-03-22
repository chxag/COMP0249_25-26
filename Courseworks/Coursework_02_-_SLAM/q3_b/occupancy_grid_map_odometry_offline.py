import sys
import math
import argparse
import os
import pygame
import numpy as np

from rplidar_driver import LidarDriver

SENSOR_MAX_RANGE_MM = 12000.0

# Display & Map Settings
WINDOW_SIZE = (800, 800)
# MAP_DIM = 800
CELL_SIZE_MM = 50

# SLAM / Probability Constants
CONFIDENCE_FREE = (10, 10, 10)
CONFIDENCE_OCCUPIED = (50, 50, 50)

# Blind Spot (User Location) - 90 degrees behind sensor
CUT_ANGLE_MIN = 135.0
CUT_ANGLE_MAX = 225.0


class PoseEstimator:
    def __init__(self, map_dim, cell_size_mm, start_x_mm=None, start_y_mm=None):
        self.map_w = map_dim
        self.map_h = map_dim
        self.cell_size = cell_size_mm
        # Default start is centre of map if not specified
        self.start_x_mm = start_x_mm if start_x_mm is not None else (map_dim * cell_size_mm) / 2
        self.start_y_mm = start_y_mm if start_y_mm is not None else (map_dim * cell_size_mm) / 2
        self.reset()

    def reset(self):
        self.x = self.start_x_mm
        self.y = self.start_y_mm
        self.theta = 0.0

    def get_pose(self):
        return self.x, self.y, self.theta

    def optimize_pose(self, scan_points, grid_map, iterations=10):
        if len(scan_points) == 0:
            return

        scan_arr = np.array(scan_points, dtype=np.float32)
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


def build_output_surface(view_surface, trajectory_points, rob_px, rob_py, curr_th):
    output = view_surface.copy()

    if len(trajectory_points) > 1:
        pygame.draw.lines(output, (0, 191, 255), False, trajectory_points, 2)

    pygame.draw.circle(output, (255, 0, 0), (rob_px, rob_py), 8)

    end_x = rob_px + 20 * math.cos(curr_th)
    end_y = rob_py + 20 * math.sin(curr_th)
    pygame.draw.line(output, (255, 0, 0), (rob_px, rob_py), (end_x, end_y), 3)

    blind_p1_x = rob_px + 30 * math.cos(curr_th + math.radians(CUT_ANGLE_MIN))
    blind_p1_y = rob_py + 30 * math.sin(curr_th + math.radians(CUT_ANGLE_MIN))
    blind_p2_x = rob_px + 30 * math.cos(curr_th + math.radians(CUT_ANGLE_MAX))
    blind_p2_y = rob_py + 30 * math.sin(curr_th + math.radians(CUT_ANGLE_MAX))

    pygame.draw.line(output, (255, 255, 0), (rob_px, rob_py), (blind_p1_x, blind_p1_y), 1)
    pygame.draw.line(output, (255, 255, 0), (rob_px, rob_py), (blind_p2_x, blind_p2_y), 1)

    return output
def run_fixed_map_slam(
    json_file_path,
    max_range_mm,
    beam_step,
    scan_skip,
    save_path,
    cell_size_mm,
    map_dim,
    start_x_mm,
    start_y_mm,
):
    pygame.init()
    screen = pygame.display.set_mode((map_dim, map_dim))
    pygame.display.set_caption("Lidar SLAM: Fixed Map + Trajectory (Offline Replay)")

    view_surface = pygame.Surface((map_dim, map_dim))
    view_surface.fill((128, 128, 128))

    occupancy_grid = np.full((map_dim, map_dim), 0.5, dtype=np.float32)
    trajectory_points = []

    estimator = PoseEstimator(map_dim, cell_size_mm, start_x_mm, start_y_mm)
    lidar = LidarDriver(mode="replay", filename=json_file_path)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    last_rob_px = int(start_x_mm / cell_size_mm)
    last_rob_py = int(start_y_mm / cell_size_mm)
    last_curr_th = 0.0

    print(f"Starting offline replay from: {json_file_path}")
    print(
        f"Settings: max_range_mm={max_range_mm}, beam_step={beam_step}, "
        f"scan_skip={scan_skip}, cell_size_mm={cell_size_mm}, "
        f"map_dim={map_dim}, start=({start_x_mm},{start_y_mm})mm"
    )

    scan_idx = 0

    try:
        for scan in lidar.iter_scans():
            scan_idx += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
                    if event.key == pygame.K_r:
                        print("RESETTING MAP...")
                        view_surface.fill((128, 128, 128))
                        occupancy_grid.fill(0.5)
                        trajectory_points.clear()
                        estimator.reset()
                        last_rob_px = int(start_x_mm / cell_size_mm)
                        last_rob_py = int(start_y_mm / cell_size_mm)
                        last_curr_th = 0.0

            if scan_skip > 1 and ((scan_idx - 1) % scan_skip != 0):
                continue

            valid_points = []
            filtered_scan = scan[::beam_step] if beam_step > 1 else scan

            for (_, angle, distance) in filtered_scan:
                if CUT_ANGLE_MIN <= angle <= CUT_ANGLE_MAX:
                    continue
                if distance <= 0:
                    continue
                if distance > max_range_mm:
                    continue
                valid_points.append((math.radians(angle), distance))

            if not valid_points:
                continue

            estimator.optimize_pose(valid_points, occupancy_grid, iterations=10)
            curr_x, curr_y, curr_th = estimator.get_pose()

            rob_px = int(curr_x / cell_size_mm)
            rob_py = int(curr_y / cell_size_mm)
            trajectory_points.append((rob_px, rob_py))

            last_rob_px = rob_px
            last_rob_py = rob_py
            last_curr_th = curr_th

            flash_surface = pygame.Surface((map_dim, map_dim))
            flash_surface.fill((0, 0, 0))
            hits_surface = pygame.Surface((map_dim, map_dim))
            hits_surface.fill((0, 0, 0))

            math_hits_x = []
            math_hits_y = []

            cos_th = math.cos(curr_th)
            sin_th = math.sin(curr_th)

            for (angle_rad, dist) in valid_points:
                lx = dist * math.cos(angle_rad)
                ly = dist * math.sin(angle_rad)

                gx_mm = (lx * cos_th - ly * sin_th) + curr_x
                gy_mm = (lx * sin_th + ly * cos_th) + curr_y

                px = int(gx_mm / cell_size_mm)
                py = int(gy_mm / cell_size_mm)

                if 0 <= px < map_dim and 0 <= py < map_dim:
                    pygame.draw.line(flash_surface, CONFIDENCE_FREE, (rob_px, rob_py), (px, py), 2)
                    pygame.draw.circle(hits_surface, CONFIDENCE_OCCUPIED, (px, py), 2)
                    math_hits_x.append(px)
                    math_hits_y.append(py)

            view_surface.blit(flash_surface, (0, 0), special_flags=pygame.BLEND_ADD)
            view_surface.blit(hits_surface, (0, 0), special_flags=pygame.BLEND_SUB)

            if math_hits_x:
                rows = np.array(math_hits_x)
                cols = np.array(math_hits_y)
                occupancy_grid[rows, cols] = np.minimum(1.0, occupancy_grid[rows, cols] + 0.1)

            output_surface = build_output_surface(
                view_surface, trajectory_points, rob_px, rob_py, curr_th,
            )

            screen.fill((40, 40, 40))
            screen.blit(output_surface, (0, 0))

            fps = int(clock.get_fps())
            info = (
                f"FPS: {fps} | Pose: {curr_x:.0f}, {curr_y:.0f} | "
                f"range={max_range_mm:.0f} beam={beam_step} cell={cell_size_mm}mm skip={scan_skip}"
            )
            text = font.render(info, True, (0, 255, 0))
            screen.blit(text, (10, 10))

            pygame.display.flip()
            clock.tick(60)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            final_surface = build_output_surface(
                view_surface, trajectory_points, last_rob_px, last_rob_py, last_curr_th,
            )
            pygame.image.save(final_surface, save_path)
            print(f"Saved map image to: {save_path}")

            npy_path = os.path.splitext(save_path)[0] + ".npy"
            np.save(npy_path, occupancy_grid)
            print(f"Saved occupancy grid to: {npy_path}")

        lidar.disconnect()
        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("--max-range-mm", type=float, default=4000.0)
    parser.add_argument("--beam-step", type=int, default=1)
    parser.add_argument("--scan-skip", type=int, default=1)
    parser.add_argument("--save", type=str, default="output/map.png")
    parser.add_argument("--cell-size-mm", type=float, default=50.0)
    parser.add_argument("--map-dim", type=int, default=800)
    parser.add_argument("--start-x-mm", type=float, default=None)
    parser.add_argument("--start-y-mm", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cell = args.cell_size_mm
    map_dim = args.map_dim
    # Default start to centre if not provided
    start_x = args.start_x_mm if args.start_x_mm is not None else (map_dim * cell) / 2
    start_y = args.start_y_mm if args.start_y_mm is not None else (map_dim * cell) / 2

    run_fixed_map_slam(
        json_file_path=args.json_file,
        max_range_mm=args.max_range_mm,
        beam_step=max(1, args.beam_step),
        scan_skip=max(1, args.scan_skip),
        save_path=args.save,
        cell_size_mm=cell,
        map_dim=map_dim,
        start_x_mm=start_x,
        start_y_mm=start_y,
    )
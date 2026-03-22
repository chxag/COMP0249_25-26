import subprocess
import os

SENSOR_MAX_RANGE_MM = 12000.0
JSON_FILE = "indoor_big_loop/indoor_big_loop.json"
DEFAULT_CELL_SIZE_MM = 50

# Physical coverage of the map in mm, and robot start position in mm
MAP_SIZE_MM = 40000       # 40 metres coverage
START_X_MM  = 20000       # start in the middle
START_Y_MM  = 20000

experiments = [
    # 1) Maximum Range
    {"title": "Range test - 4m", "max_range_mm": 4000.0, "beam_step": 1, "cell_size_mm": DEFAULT_CELL_SIZE_MM, "scan_skip": 1, "save": "indoor_big_loop/occupancy_grid/occ_range_4m.png"},
    {"title": "Range test - sensor max",  "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "cell_size_mm": DEFAULT_CELL_SIZE_MM, "scan_skip": 1, "save": "indoor_big_loop/occupancy_grid/occ_range_12m.png"},

    # 2) Angular Resolution
    {"title": "Angular resolution - full scan","max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "cell_size_mm": DEFAULT_CELL_SIZE_MM, "scan_skip": 1, "save": "indoor_big_loop/occupancy_grid/occ_angular_full.png"},
    {"title": "Angular resolution - every 2nd beam", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 2, "cell_size_mm": DEFAULT_CELL_SIZE_MM, "scan_skip": 1, "save": "indoor_big_loop/occupancy_grid/occ_angular_n2.png"},
    {"title": "Angular resolution - every 3rd beam", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 3, "cell_size_mm": DEFAULT_CELL_SIZE_MM, "scan_skip": 1, "save": "indoor_big_loop/occupancy_grid/occ_angular_n3.png"},

    # 3) Cell size (occupancy grid resolution)
    {"title": "Cell size - 50mm (coarse)", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "cell_size_mm": 50, "scan_skip": 1, "save": "indoor_big_loop/occupancy_grid/occ_cell_50mm.png"},
    {"title": "Cell size - 25mm (fine)",   "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "cell_size_mm": 25, "scan_skip": 1, "save": "indoor_big_loop/occupancy_grid/occ_cell_25mm.png"},

    # 4) Scan Rate
    {"title": "Scan rate - full", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "cell_size_mm": DEFAULT_CELL_SIZE_MM, "scan_skip": 1, "save": "indoor_big_loop/occupancy_grid/occ_scanrate_full.png"},
    {"title": "Scan rate - 50% reduction", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "cell_size_mm": DEFAULT_CELL_SIZE_MM, "scan_skip": 2, "save": "indoor_big_loop/occupancy_grid/occ_scanrate_50.png"},
    {"title": "Scan rate - keep 1 in 3",   "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "cell_size_mm": DEFAULT_CELL_SIZE_MM, "scan_skip": 3, "save": "indoor_big_loop/occupancy_grid/occ_scanrate_33.png"},
]

for exp in experiments:
    cell = exp["cell_size_mm"]
    map_dim   = MAP_SIZE_MM // cell
    start_x   = START_X_MM 
    start_y   = START_Y_MM

    os.makedirs(os.path.dirname(exp["save"]), exist_ok=True)

    cmd = [
        "python", "occupancy_grid_map_odometry_offline.py", JSON_FILE,
        "--max-range-mm",  str(exp["max_range_mm"]),
        "--beam-step",     str(exp["beam_step"]),
        "--scan-skip",     str(exp["scan_skip"]),
        "--cell-size-mm",  str(cell),
        "--map-dim",       str(map_dim),
        "--start-x-mm",    str(start_x),
        "--start-y-mm",    str(start_y),
        "--save",          exp["save"],
    ]

    print(f"Running: {exp['title']}  (map_dim={map_dim}, cell={cell}mm)")
    subprocess.run(cmd, check=True)
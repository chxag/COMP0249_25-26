import os
from rplidar_icp_offlne import load_scans_from_json, run_offline_slam, plot_result

SENSOR_MAX_RANGE_MM = 12000.0
JSON_FILE = "outdoor_area/charvi_outside.json"

experiments = [
    # 1) Maximum Range
    {"title": "Range test - 4m", "max_range_mm": 4000.0, "beam_step": 1, "scan_skip": 1, "save": "outdoor_area/point_cloud/range_4m.png"},
    {"title": "Range test - sensor max", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "scan_skip": 1, "save": "outdoor_area/point_cloud/range_max.png"},

    # 2) Angular Resolution
    {"title": "Angular resolution - full scan", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "scan_skip": 1, "save": "outdoor_area/point_cloud/angular_full.png"},
    {"title": "Angular resolution - every 2nd beam", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 2, "scan_skip": 1, "save": "outdoor_area/point_cloud/angular_n2.png"},
    {"title": "Angular resolution - every 3rd beam", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 3, "scan_skip": 1, "save": "outdoor_area/point_cloud/angular_n3.png"},

    # 3) Scan Rate
    {"title": "Scan rate - full", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "scan_skip": 1, "save": "outdoor_area/point_cloud/scanrate_full.png"},
    {"title": "Scan rate - 50% reduction", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "scan_skip": 2, "save": "outdoor_area/point_cloud/scanrate_50.png"},
    {"title": "Scan rate - keep 1 in 3", "max_range_mm": SENSOR_MAX_RANGE_MM, "beam_step": 1, "scan_skip": 3, "save": "outdoor_area/point_cloud/scanrate_33.png"},
]

if __name__ == "__main__":
    scans = load_scans_from_json(JSON_FILE)

    for exp in experiments:
        print(f"Running: {exp['title']}")
        result = run_offline_slam(
            scans,
            max_range_mm=exp["max_range_mm"],
            beam_step=exp["beam_step"],
            scan_skip=exp["scan_skip"],
            title=exp["title"],
        )

        if result[0] is None:
            print(f"  Failed: {exp['title']}")
            continue

        map_points, trajectory = result
        plot_result(map_points, trajectory, exp["title"], exp["save"])
        print(f"  Saved: {exp['save']}")
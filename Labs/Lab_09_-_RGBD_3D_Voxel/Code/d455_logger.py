import pyrealsense2 as rs
import numpy as np
import cv2
import os
import csv
from datetime import datetime

# Base directory for all recordings
base_path = 'captures'
if not os.path.exists(base_path):
    os.makedirs(base_path)

# Configure D455

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

print("Streaming... Press 'SPACE' to capture, 'ESC' to stop.")

# Setup the Align Object
align_to = rs.stream.color
align = rs.align(align_to)

current_session_folder = ""
session_name = ""

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Display (Colorized Depth + RGB)
        depth_raw = np.asanyarray(depth_frame.get_data())
        color_raw = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('D455 Capture', np.hstack((color_raw, depth_colormap)))

        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            if session_name == "": 
                session_name = datetime.now().strftime("Session_%Y%m%d_%H%M%S")
                current_session_folder = os.path.join(base_path, session_name)
                os.makedirs(current_session_folder)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # Save RGB
            cv2.imwrite(os.path.join(current_session_folder, f'img_{timestamp}.png'), color_raw)
            
            # Save Raw Depth
            np.save(os.path.join(current_session_folder, f'depth_{timestamp}.npy'), depth_raw)
            
            # Log Metadata to CSV
            meta_path = os.path.join(current_session_folder, f'meta_{timestamp}.csv')
            with open(meta_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Field', 'Value'])
                
                # Standard Metadatas
                writer.writerow(['Frame_Number', color_frame.get_frame_number()])
                writer.writerow(['System_Timestamp', timestamp])

                # Hardware Specific Metadata (Exposure, Gain, etc.)
                for opt in [rs.frame_metadata_value.actual_exposure, 
                            rs.frame_metadata_value.gain_level,
                            rs.frame_metadata_value.frame_timestamp]:
                    if color_frame.supports_frame_metadata(opt):
                        writer.writerow([str(opt).split('.')[-1], color_frame.get_frame_metadata(opt)])

            print(f"Captured: {timestamp}")

        elif key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
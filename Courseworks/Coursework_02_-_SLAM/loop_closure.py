'''
We will follow the theory from the slides and lecture 6.

Based on that, the steps to build a loop closure mechanism for the LiDAR SLAM task include:

A) Maintain a Spatial and Temporal History: We must keep a historical record of all past robot poses (vertices) and their corresponding laser scans as the robot navigates, while ignoring the most recently collected frames (temporal filtering) to avoid trivial sequential matching.
B) Proximity-Based Trigger (KD-Tree): Every time we receive a new scan, instead of comparing it against the entire history, we perform a fast spatial search using a KD-Tree. We query the tree to find any historical poses that fall within a specific physical radius (e.g., 4.25 meters) of our current estimated position.
C) Geometric Verification (ICP): For any historical candidates found in the spatial radius, we evaluate how well the current raw laser scan aligns with the historical laser scan using Iterative Closest Point (ICP) matching. This verifies if the actual environment geometry matches, rather than just the estimated coordinates.
D) False Positive Rejection and Constraint Addition: We evaluate the mean distance error from the ICP alignment. If the error is below a strict threshold (e.g., 0.15), we officially declare a loop closure. We then add this match as a new constraint (edge) into the back-end graph optimizer, weighted by the inverse of the ICP covariance matrix, to correct the accumulated drift.
'''

#source for the main code: https://github.com/HobbySingh/Graph-SLAM


import numpy as np
import scipy.spatial
import time

import icp 
from pose_se2 import PoseSE2

class LoopClosureDetector:
    def __init__(self, search_radius=4.25, icp_error_thresh=0.15, temporal_skip=50):
        """
        Initializes the Loop Closure Detector.
        
        :param search_radius: Maximum distance (meters) to search for nearby historical poses.
        :param icp_error_thresh: Maximum allowable mean ICP error to accept the loop closure.
        :param temporal_skip: Minimum number of frames to skip to avoid matching with the immediate past.
        """
        self.search_radius = search_radius
        self.icp_error_thresh = icp_error_thresh
        self.temporal_skip = temporal_skip

    def find_loop_closure(self, curr_pose, curr_idx, all_lasers, pose_graph):
        """
        Detects loop closures and adds the corresponding edge between the current
        pose and detected loop closure pose.
        
        :param curr_pose: The current PoseSE2 of the robot
        :param curr_idx: The current vertex index in the graph
        :param all_lasers: List of all laser scans recorded so far
        :param pose_graph: The pose graph optimizer object
        """
        # 1. Temporal Filtering: Avoid checking poses that are too recent
        if curr_idx <= self.temporal_skip:
            return False
        
        searchable_indices = range(curr_idx - self.temporal_skip)
        
        # Extract historical poses up to the temporal skip
        poses = np.array([pose_graph.get_pose(idx).arr[0:2] for idx in searchable_indices]).squeeze()
        
        # 2. Spatial Proximity Trigger: KD-Tree search for fast radius queries
        kd_tree = scipy.spatial.cKDTree(poses)
        
        # Query the KD-tree for points within 'search_radius' of the current position
        curr_xy = curr_pose.arr[0:2].T
        candidate_idxs = kd_tree.query_ball_point(curr_xy, self.search_radius)[0]

        loop_found = False

        # 3. Geometric Verification: ICP matching
        for i in candidate_idxs:
            with np.errstate(all="raise"):
                try:
                    # Perform ICP between the candidate historical scan and the current scan
                    # Initial pose guess is set to Identity because we are comparing relative shapes
                    tf, distances, iterations, cov = icp.icp(
                        A=all_lasers[i], 
                        B=all_lasers[curr_idx], 
                        init_pose=np.eye(3), 
                        max_iterations=80, 
                        tolerance=0.0001
                    )
                except Exception as e:
                    print(f"[Loop Closure] ICP Exception at index {i}: {e}")
                    continue

            # 4. False Positive Rejection
            mean_error = np.mean(distances)
            if mean_error < self.icp_error_thresh:
                print(f"[Loop Closure] Success! Matched frame {curr_idx} with frame {i}. Error: {mean_error:.3f}")
                
                # Add the loop closure constraint to the Pose Graph
                # The information matrix is the inverse of the ICP covariance
                information_matrix = np.linalg.inv(cov)
                
                pose_graph.add_edge(
                    vertices=[curr_idx, i], 
                    measurement=PoseSE2.from_rt_matrix(tf), 
                    information=information_matrix
                )
                loop_found = True
                
                # Optional: Artificial delay for visualization synchronization (e.g., ROS bag or PyGame execution)
                time.sleep(0.05)
                
        return loop_found
    
    '''
    Report:
    
    Detection Trigger Logic:

    The robot constantly receives new laser scans as it navigates. For each new scan, the loop closure detector is invoked with the current pose and scan index. The first step is temporal filtering, where we ignore any historical poses that are too recent (within the last 50 frames) to avoid trivial matches with immediately preceding scans.
    Next, we extract the 2D positions of all historical poses up to the temporal skip and build a KD-tree for efficient spatial querying. We query this tree to find any historical poses that fall within a 4.25-meter radius of the current estimated position. This spatial proximity trigger significantly reduces the number of candidate scans we need to evaluate, focusing only on those that are physically nearby.
    '''
"""
Module 1 — Sensor Data Acquisition & Preprocessing Node

Simulated sensor suite publishing to ROS-style topics:
- /camera/image — RGB camera frames (visual data Xv)
- /odom/pose — Robot pose/kinematics (Xp)
- /lidar/scan — Occupancy/range data (Xo)

Preprocessing applied to visual frames:
- Resize to fixed resolution (256×256)
- Normalize pixel values to [0, 1]
- Noise reduction for agricultural illumination variation
"""

import numpy as np
import cv2
import time
import logging
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.node_base import NodeBase

logger = logging.getLogger(__name__)


# Topic names (mimicking ROS topic conventions)
TOPIC_CAMERA = '/camera/image'
TOPIC_ODOM = '/odom/pose'
TOPIC_LIDAR = '/lidar/scan'


class SensorNode(NodeBase):
    """
    Sensor acquisition and preprocessing node.
    
    Acquires sensor data from the simulation (or real hardware) and
    publishes preprocessed streams. Implements the sensor input set:
    X = {Xv, Xp, Xo} where:
    - Xv = visual frames (preprocessed RGB)
    - Xp = robot pose/kinematics
    - Xo = occupancy/range data from LiDAR
    """
    
    def __init__(self, farm_world, robot, target_resolution=(256, 256),
                 rate_hz: float = 30.0):
        super().__init__('sensor_node', rate_hz=rate_hz)
        
        self.farm_world = farm_world
        self.robot = robot
        self.target_res = target_resolution
        
        # Camera parameters
        self.fov_width = 3.0   # meters
        self.fov_depth = 5.0   # meters
        
        # LiDAR parameters
        self.lidar_rays = 360
        self.lidar_range = 10.0
        
        # Frame counter
        self.frame_id = 0
    
    def on_start(self):
        """Set up latched topics."""
        self.bus.set_latch(TOPIC_ODOM)
        logger.info("Sensor node started — publishing camera, odom, lidar")
    
    def on_update(self, dt: float):
        """Acquire and publish all sensor data."""
        x, y, theta = self.robot.get_pose()
        odom_x, odom_y, odom_theta = self.robot.get_odom_pose()
        
        # --- Publish Odometry (Xp) ---
        odom_msg = {
            'x': odom_x,
            'y': odom_y,
            'theta': odom_theta,
            'gt_x': x,  # Ground truth for evaluation
            'gt_y': y,
            'gt_theta': theta,
            'v_linear': self.robot.state.v_linear,
            'v_angular': self.robot.state.v_angular,
            'timestamp': self.robot.state.timestamp,
            'frame_id': self.frame_id
        }
        self.publish(TOPIC_ODOM, odom_msg)
        
        # --- Acquire and preprocess camera frame (Xv) ---
        # Get simulated detections from farm world
        detections = self.farm_world.get_camera_detections(
            x, y, self.fov_width, self.fov_depth, theta
        )
        
        # Generate synthetic camera image
        camera_frame = self._generate_camera_frame(x, y, theta)
        
        # Preprocess: resize, normalize, denoise
        processed_frame = self._preprocess_frame(camera_frame)
        
        camera_msg = {
            'frame': processed_frame,
            'raw_frame': camera_frame,
            'detections_gt': detections,  # Ground truth for training
            'timestamp': self.robot.state.timestamp,
            'frame_id': self.frame_id,
            'robot_pose': (x, y, theta)
        }
        self.publish(TOPIC_CAMERA, camera_msg)
        
        # --- Acquire LiDAR scan (Xo) ---
        ranges = self.farm_world.get_lidar_scan(
            x, y, theta, self.lidar_rays, self.lidar_range
        )
        
        lidar_msg = {
            'ranges': ranges,
            'angle_min': 0.0,
            'angle_max': 2 * np.pi,
            'angle_increment': 2 * np.pi / self.lidar_rays,
            'range_max': self.lidar_range,
            'timestamp': self.robot.state.timestamp,
            'frame_id': self.frame_id
        }
        self.publish(TOPIC_LIDAR, lidar_msg)
        
        self.frame_id += 1
    
    def _generate_camera_frame(self, x: float, y: float, theta: float) -> np.ndarray:
        """
        Generate a synthetic camera frame from the farm world.
        
        Creates a top-down view of the field in front of the robot,
        showing crops (green) and weeds (various colors on green background).
        """
        h, w = self.target_res
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background: soil/field (brown-green)
        frame[:, :] = [34, 85, 34]  # Dark green base
        
        # Add field texture noise
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Draw crop rows as lighter green stripes
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        for patch in self.farm_world.weed_patches:
            dx = patch.center_x - x
            dy = patch.center_y - y
            
            local_x = dx * cos_t + dy * sin_t
            local_y = -dx * sin_t + dy * cos_t
            
            if 0 < local_x < self.fov_depth and abs(local_y) < self.fov_width / 2:
                # Map to pixel coordinates
                px = int((0.5 + local_y / self.fov_width) * w)
                py = int((local_x / self.fov_depth) * h)
                radius_px = int(patch.radius / self.fov_width * w * 0.3)
                
                if 0 <= px < w and 0 <= py < h:
                    # Draw weed patch as colored blob
                    color = self._species_color(patch.species_id)
                    cv2.circle(frame, (px, py), max(3, radius_px), color, -1)
        
        return frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess camera frame:
        1. Resize to target resolution
        2. Apply Gaussian blur for noise reduction
        3. Normalize pixel values to [0, 1]
        """
        # Resize
        resized = cv2.resize(frame, self.target_res)
        
        # Gaussian blur for noise reduction (agricultural illumination)
        denoised = cv2.GaussianBlur(resized, (3, 3), 0.5)
        
        # Normalize to [0, 1] float32
        normalized = denoised.astype(np.float32) / 255.0
        
        return normalized
    
    @staticmethod
    def _species_color(species_id: int) -> tuple:
        """Get a distinctive color for each weed species."""
        colors = [
            (0, 0, 180),    # Red-ish
            (180, 0, 180),  # Purple
            (0, 140, 255),  # Orange
            (0, 255, 255),  # Yellow
            (255, 0, 0),    # Blue
            (0, 200, 0),    # Green variant
            (200, 200, 0),  # Cyan
            (100, 0, 200),  # Dark red
            (50, 180, 50),  # Lime
        ]
        return colors[species_id % len(colors)]

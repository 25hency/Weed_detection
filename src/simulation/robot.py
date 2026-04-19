"""
Robot Model — Differential Drive Agricultural Robot (AFRS Paper §3.2)

Key constants (paper §3.2):
    ROBOT_SPEED    = 0.5 m/s
    TRAVERSAL_TIME = CELL_SIZE / ROBOT_SPEED = 0.5 / 0.5 = 1.0 s per cell

Odometry reports position in METRES (x = col × CELL_SIZE, y = row × CELL_SIZE).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

from simulation.farm_world import CELL_SIZE

# ── Physical constants (paper §3.2) ──────────────────────────────────────────
ROBOT_SPEED     = 0.5   # m/s
TRAVERSAL_TIME  = CELL_SIZE / ROBOT_SPEED   # 1.0 s per cell


@dataclass
class RobotState:
    """Ground-truth robot state (all positions in metres)."""
    x:         float = 0.5   # metres (column 0, centre = 0.5)
    y:         float = 0.5   # metres (row 0, centre = 0.5)
    theta:     float = 0.0   # heading in radians
    v_linear:  float = 0.0   # m/s
    v_angular: float = 0.0   # rad/s
    timestamp: float = 0.0   # sim time (s)


class DifferentialDriveRobot:
    """
    Differential-drive agricultural robot with simulated sensor suite.

    Sensors:
        RGB Camera  — via FarmWorld.get_camera_detections()
        2D LiDAR    — via FarmWorld.get_lidar_scan()
        Odometry    — pose estimation with Gaussian drift

    Speed: ROBOT_SPEED = 0.5 m/s (paper §3.2).
    Traversal time per 0.5 m cell: TRAVERSAL_TIME = 1.0 s.
    All position values reported in METRES.
    """

    def __init__(
        self,
        start_x: float = 0.5,
        start_y: float = 0.5,
        start_theta: float = 0.0,
        max_linear_vel: float = ROBOT_SPEED,
        max_angular_vel: float = 1.0,
        wheel_base: float = 0.3,
        odom_noise_linear: float = 0.005,
        odom_noise_angular: float = 0.003,
    ):
        self.state = RobotState(x=start_x, y=start_y, theta=start_theta)
        self.max_v = max_linear_vel
        self.max_w = max_angular_vel
        self.wheel_base = wheel_base
        self.odom_noise_lin = odom_noise_linear
        self.odom_noise_ang = odom_noise_angular

        # Odometry estimate (accumulates drift)
        self.odom_x     = start_x
        self.odom_y     = start_y
        self.odom_theta = start_theta

        # Path history for visualisation and length computation (metres)
        self.path_history: list = [(start_x, start_y)]

        # Sprayer
        self.sprayer_active   = False
        self.total_spray_time = 0.0
        self.spray_count      = 0
        self.spray_events: list = []

        logger.info(
            f"Robot initialised at ({start_x:.2f} m, {start_y:.2f} m) | "
            f"speed={ROBOT_SPEED} m/s | traversal_time={TRAVERSAL_TIME} s/cell"
        )

    # ── Kinematics ────────────────────────────────────────────────────────────

    def update(self, dt: float, v_cmd: float = 0.0, w_cmd: float = 0.0) -> None:
        """
        Advance robot state by dt seconds.

        Args:
            dt    : time step (seconds)
            v_cmd : commanded linear velocity (m/s), clamped to ±ROBOT_SPEED
            w_cmd : commanded angular velocity (rad/s)
        """
        v = np.clip(v_cmd, -self.max_v, self.max_v)
        w = np.clip(w_cmd, -self.max_w, self.max_w)

        self.state.v_linear  = v
        self.state.v_angular = w
        self.state.timestamp += dt

        # Ground-truth kinematics (perfect)
        self.state.x     += v * np.cos(self.state.theta) * dt
        self.state.y     += v * np.sin(self.state.theta) * dt
        self.state.theta  = self._norm_angle(self.state.theta + w * dt)

        # Odometry with Gaussian noise (drift)
        v_n = v + np.random.normal(0, self.odom_noise_lin)
        w_n = w + np.random.normal(0, self.odom_noise_ang)
        self.odom_x     += v_n * np.cos(self.odom_theta) * dt
        self.odom_y     += v_n * np.sin(self.odom_theta) * dt
        self.odom_theta  = self._norm_angle(self.odom_theta + w_n * dt)

        self.path_history.append((self.state.x, self.state.y))
        if self.sprayer_active:
            self.total_spray_time += dt

    # ── Pose accessors ────────────────────────────────────────────────────────

    def get_pose(self) -> Tuple[float, float, float]:
        """Return ground-truth pose (x_m, y_m, theta_rad)."""
        return self.state.x, self.state.y, self.state.theta

    def get_odom_pose(self) -> Tuple[float, float, float]:
        """Return odometry-estimated pose with accumulated drift (metres)."""
        return self.odom_x, self.odom_y, self.odom_theta

    def get_grid_cell(self, cell_size: float = CELL_SIZE) -> Tuple[int, int]:
        """Return (row, col) grid cell of current ground-truth position."""
        col = int(self.state.x / cell_size)
        row = int(self.state.y / cell_size)
        return row, col

    # ── Sprayer control ──────────────────────────────────────────────────────

    def activate_sprayer(self, confidence: float, density: float) -> None:
        if not self.sprayer_active:
            self.sprayer_active = True
            self.spray_count   += 1
            self.spray_events.append({
                "x":          self.state.x,
                "y":          self.state.y,
                "time":       self.state.timestamp,
                "confidence": confidence,
                "density":    density,
            })

    def deactivate_sprayer(self) -> None:
        self.sprayer_active = False

    # ── Path length ──────────────────────────────────────────────────────────

    def get_path_length(self) -> float:
        """Total Euclidean path length in METRES."""
        total = 0.0
        for i in range(1, len(self.path_history)):
            x1, y1 = self.path_history[i - 1]
            x2, y2 = self.path_history[i]
            total += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return total

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _norm_angle(a: float) -> float:
        while a >  np.pi: a -= 2 * np.pi
        while a < -np.pi: a += 2 * np.pi
        return a

"""
Module 3 — Weed Infestation Heatmap Generator  (AFRS Paper §3.4)

Heatmap equation (paper eq. 6):
    H(i,j)_t = δ · H(i,j)_{t-1} + Σ_k  c_k · I(i,j,k)

Decay factor (paper §3.4 — corrected to match "stale cell < 0.05 in 20 steps"):
    δ = 0.85   →  0.85^20 ≈ 0.039  < 0.05  ✓
    (The original δ = 0.95 requires ~58 steps, contradicting the paper claim.)

Gaussian spreading kernel — 3×3, sigma = 0.8.
    Each detection spreads to all 8 neighbours proportionally.

Normalisation — heatmap is normalised to [0, 1] after every update.

Crop-class detections (class_id == 15) are EXCLUDED from the heatmap.

Dynamic replanning is triggered when the sum of heatmap changes in the
forward-path region exceeds delta_threshold = 0.15.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.node_base import NodeBase
from simulation.farm_world import GRID_SIZE, CELL_SIZE

logger = logging.getLogger(__name__)

TOPIC_DETECTIONS = '/detection/weeds'
TOPIC_ODOM       = '/odom/pose'
TOPIC_HEATMAP    = '/heatmap/grid'

# ── Named constants (paper §3.4) ─────────────────────────────────────────────
DECAY_FACTOR          = 0.85    # δ  (0.85^20 ≈ 0.039 < 0.05 → stale in 20 steps)
GAUSSIAN_SIGMA        = 0.8     # kernel sigma for 3×3 spreading
REPLAN_DELTA_THRESHOLD = 0.15   # sum-of-changes threshold to trigger replanning
CLASS_ID_CROP         = 15      # must NOT contribute to heatmap


def _make_gaussian_kernel(sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """Build a normalised 3×3 Gaussian kernel."""
    k = np.zeros((3, 3), dtype=np.float64)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            k[dr + 1, dc + 1] = np.exp(-(dr**2 + dc**2) / (2 * sigma**2))
    # Zero the centre (centre accumulation is done separately)
    k[1, 1] = 0.0
    # Normalise neighbour weights to sum to 1
    s = k.sum()
    if s > 0:
        k /= s
    return k.astype(np.float32)


_GAUSS_KERNEL = _make_gaussian_kernel(GAUSSIAN_SIGMA)


class HeatmapNode(NodeBase):
    """
    Weed Infestation Heatmap Generator.

    Grid dimensions match FarmWorld: GRID_SIZE × GRID_SIZE (20 × 20).
    Decay factor δ = DECAY_FACTOR = 0.85 (paper §3.4, corrected).
    """

    def __init__(
        self,
        grid_size: int   = GRID_SIZE,
        resolution: float = CELL_SIZE,
        decay_factor: float = DECAY_FACTOR,
        rate_hz: float = 10.0,
    ):
        super().__init__('heatmap_node', rate_hz=rate_hz)

        self.grid_size   = grid_size
        self.grid_w      = grid_size
        self.grid_h      = grid_size
        self.resolution  = resolution
        self.decay_factor = decay_factor          # δ = 0.85

        # H(i,j) — the accumulated, normalised heatmap
        self.heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Running maximum (used for normalisation)
        self._running_max = 1e-9

        # Tracking
        self.total_detections  = 0
        self.update_times: List[float] = []

        # Replanning trigger state
        self.prev_heatmap   = np.zeros_like(self.heatmap)
        self.heatmap_changed = False
        self.forward_delta   = 0.0

        self.latest_detection_msg = None
        self.latest_odom_msg      = None

    # ── Node lifecycle ────────────────────────────────────────────────────────

    def on_start(self):
        self.subscribe(TOPIC_DETECTIONS, self._on_detection)
        self.subscribe(TOPIC_ODOM,       self._on_odom)
        self.bus.set_latch(TOPIC_HEATMAP)
        logger.info(
            f"HeatmapNode started — grid {self.grid_size}×{self.grid_size}, "
            f"δ={self.decay_factor}, σ={GAUSSIAN_SIGMA}"
        )

    def _on_detection(self, msg): self.latest_detection_msg = msg
    def _on_odom(self, msg):      self.latest_odom_msg = msg

    # ── Main update ──────────────────────────────────────────────────────────

    def on_update(self, dt: float):
        if self.latest_detection_msg is None:
            return

        t_start        = time.perf_counter()
        detection_msg  = self.latest_detection_msg
        self.latest_detection_msg = None

        prev = self.heatmap.copy()

        # ── Step 1: Temporal decay ────────────────────────────────────────────
        self.heatmap *= self.decay_factor

        # ── Step 2: Accumulate detections (skip crop class) ──────────────────
        detections = detection_msg.get('detections', [])
        for det in detections:
            if det.get('class_id', CLASS_ID_CROP) == CLASS_ID_CROP:
                continue   # crop-class must NOT contribute to heatmap

            world_x = det.get('world_x', 0)
            world_y = det.get('world_y', 0)
            ck      = float(det.get('confidence', 0))

            col = int(world_x / self.resolution)
            row = int(world_y / self.resolution)

            if not (0 <= row < self.grid_h and 0 <= col < self.grid_w):
                continue

            # Centre accumulation: H(i,j) += c_k
            self.heatmap[row, col] += ck

            # 3×3 Gaussian neighbour spreading
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
                        weight = _GAUSS_KERNEL[dr + 1, dc + 1]
                        self.heatmap[nr, nc] += ck * weight

            self.total_detections += 1

        # ── Step 3: Normalise to [0, 1] ──────────────────────────────────────
        current_max = float(self.heatmap.max())
        if current_max > self._running_max:
            self._running_max = current_max
        if self._running_max > 1e-9:
            self.heatmap = np.clip(self.heatmap / self._running_max, 0.0, 1.0)

        # ── Step 4: Check replanning trigger ─────────────────────────────────
        delta_sum = float(np.abs(self.heatmap - prev).sum())
        self.forward_delta   = delta_sum
        self.heatmap_changed = delta_sum > REPLAN_DELTA_THRESHOLD
        self.prev_heatmap    = self.heatmap.copy()

        update_time_ms = (time.perf_counter() - t_start) * 1000
        self.update_times.append(update_time_ms)

        heatmap_msg = {
            'heatmap':         self.heatmap.copy(),
            'grid_width':      self.grid_w,
            'grid_height':     self.grid_h,
            'resolution':      self.resolution,
            'total_detections': self.total_detections,
            'heatmap_changed': self.heatmap_changed,
            'forward_delta':   self.forward_delta,
            'update_time_ms':  update_time_ms,
            'timestamp':       detection_msg.get('timestamp', 0),
            'max_density':     float(self.heatmap.max()),
            'mean_density':    float(self.heatmap.mean()),
        }
        self.publish(TOPIC_HEATMAP, heatmap_msg)

    # ── Direct update (used by SimulationEngine without message bus) ──────────

    def update_direct(self, detections: List[dict]) -> None:
        """Update heatmap directly from a list of detection dicts (no bus)."""
        prev = self.heatmap.copy()
        self.heatmap *= self.decay_factor

        for det in detections:
            if det.get('class_id', CLASS_ID_CROP) == CLASS_ID_CROP:
                continue
            col = int(det.get('world_x', 0) / self.resolution)
            row = int(det.get('world_y', 0) / self.resolution)
            ck  = float(det.get('confidence', 0))
            if not (0 <= row < self.grid_h and 0 <= col < self.grid_w):
                continue
            self.heatmap[row, col] += ck
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
                        self.heatmap[nr, nc] += ck * _GAUSS_KERNEL[dr + 1, dc + 1]
            self.total_detections += 1

        # Normalise
        current_max = float(self.heatmap.max())
        if current_max > self._running_max:
            self._running_max = current_max
        if self._running_max > 1e-9:
            self.heatmap = np.clip(self.heatmap / self._running_max, 0.0, 1.0)

        delta_sum            = float(np.abs(self.heatmap - prev).sum())
        self.forward_delta   = delta_sum
        self.heatmap_changed = delta_sum > REPLAN_DELTA_THRESHOLD

    # ── Query helpers ─────────────────────────────────────────────────────────

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        return (
            max(0, min(row, self.grid_h - 1)),
            max(0, min(col, self.grid_w - 1)),
        )

    def get_density_at(self, x: float, y: float) -> float:
        row, col = self.world_to_grid(x, y)
        return float(self.heatmap[row, col])

    def get_density_at_cell(self, row: int, col: int) -> float:
        if 0 <= row < self.grid_h and 0 <= col < self.grid_w:
            return float(self.heatmap[row, col])
        return 0.0

    def get_avg_update_time(self) -> float:
        if not self.update_times:
            return 0.0
        return float(np.mean(self.update_times[-100:]))

    def reset(self) -> None:
        """Reset heatmap (e.g. between scenario runs)."""
        self.heatmap       = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self._running_max  = 1e-9
        self.total_detections = 0
        self.update_times.clear()

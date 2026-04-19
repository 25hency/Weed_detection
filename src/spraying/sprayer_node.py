"""
Module 5 — Selective Precision Spraying Controller  (AFRS Paper §3.6)

Spray policy (paper eq. 9):
    Spray(x,y) = 1  iff  H(x,y) > τ_d  AND  c_local > τ_c

Named thresholds (paper §3.6):
    TAU_DENSITY    = 0.3
    TAU_CONFIDENCE = 0.5

Three strategies tracked simultaneously (Table 7):
    UNIFORM         — spray every visited cell
    DETECTION_ONLY  — spray when any detection present (confidence check only)
    DUAL_THRESHOLD  — spray only when H > τ_d AND max(c_k) > τ_c   [proposed]

Per-strategy metrics:
    spray_count          — number of spray activations
    weed_cells_treated   — sprays on cells that truly contain a weed
    false_spray_count    — sprays on cells with no ground-truth weed

Derived:
    chemical_pct      = spray_count / uniform_spray_count × 100
    weed_coverage_pct = weed_cells_treated / total_weed_cells × 100
    false_spray_rate  = false_spray_count / spray_count × 100

Spray area per activation = CELL_SIZE² = 0.25 m² (paper §3.6).
"""

import time
import numpy as np
import logging
from typing import List, Dict, Optional, Set, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.node_base import NodeBase
from simulation.farm_world import CELL_SIZE

logger = logging.getLogger(__name__)

TOPIC_DETECTIONS = '/detection/weeds'
TOPIC_HEATMAP    = '/heatmap/grid'
TOPIC_ODOM       = '/odom/pose'
TOPIC_SPRAY_CMD  = '/sprayer/cmd'

# ── Named threshold constants (paper §3.6) ───────────────────────────────────
TAU_DENSITY     = 0.30   # τ_d
TAU_CONFIDENCE  = 0.50   # τ_c
SPRAY_AREA_M2   = CELL_SIZE ** 2   # 0.25 m²

# Strategy labels
UNIFORM        = "uniform"
DETECTION_ONLY = "detection_only"
DUAL_THRESHOLD = "dual_threshold"


class StrategyTracker:
    """Tracks spray metrics for one strategy."""

    def __init__(self, name: str):
        self.name              = name
        self.spray_count:          int = 0
        self.weed_cells_treated:   int = 0
        self.false_spray_count:    int = 0
        self.sprayed_cells:        Set[Tuple[int, int]] = set()

    def record(self, cell: Tuple[int, int], has_weed: bool) -> None:
        if cell not in self.sprayed_cells:
            self.sprayed_cells.add(cell)
            self.spray_count += 1
            if has_weed:
                self.weed_cells_treated += 1
            else:
                self.false_spray_count  += 1

    def metrics(self, uniform_spray_count: int, total_weed_cells: int) -> dict:
        uc = max(1, uniform_spray_count)
        sc = max(1, self.spray_count)
        tw = max(1, total_weed_cells)
        return {
            "chemical_pct":      round(self.spray_count / uc * 100, 2),
            "weed_coverage_pct": round(self.weed_cells_treated / tw * 100, 2),
            "false_spray_rate_pct": round(self.false_spray_count / sc * 100, 2),
        }


class SprayerNode(NodeBase):
    """
    Selective Precision Spraying Controller.

    Tracks UNIFORM, DETECTION_ONLY, and DUAL_THRESHOLD strategies
    simultaneously to reproduce Table 7 of the paper.
    """

    def __init__(
        self,
        robot=None,
        density_threshold:    float = TAU_DENSITY,
        confidence_threshold: float = TAU_CONFIDENCE,
        spray_duration:       float = 1.0,   # seconds (= TRAVERSAL_TIME)
        flow_rate:            float = 0.10,  # L/s
        rate_hz:              float = 10.0,
        weed_ground_truth:    Optional[np.ndarray] = None,
    ):
        super().__init__('sprayer_node', rate_hz=rate_hz)

        self.robot                = robot
        self.density_threshold    = density_threshold
        self.conf_threshold       = confidence_threshold
        self.spray_duration       = spray_duration
        self.flow_rate            = flow_rate
        self.weed_ground_truth    = weed_ground_truth

        # Message state
        self.latest_detection_msg = None
        self.latest_heatmap_msg   = None
        self.latest_odom_msg      = None

        # Three strategy trackers
        self.trackers: Dict[str, StrategyTracker] = {
            UNIFORM:        StrategyTracker(UNIFORM),
            DETECTION_ONLY: StrategyTracker(DETECTION_ONLY),
            DUAL_THRESHOLD: StrategyTracker(DUAL_THRESHOLD),
        }

        # Dual-threshold spray state (active strategy for robot control)
        self.spray_active  = False
        self.spray_timer   = 0.0
        self.total_chemical = 0.0

        # All visited cells (for uniform baseline)
        self.traversed_cells: Set[Tuple[int, int]] = set()

        logger.info(
            f"SprayerNode started — τ_d={density_threshold}, τ_c={confidence_threshold}"
        )

    # ── Subscriptions ─────────────────────────────────────────────────────────

    def on_start(self):
        self.subscribe(TOPIC_DETECTIONS, self._on_detection)
        self.subscribe(TOPIC_HEATMAP,    self._on_heatmap)
        self.subscribe(TOPIC_ODOM,       self._on_odom)

    def _on_detection(self, msg): self.latest_detection_msg = msg
    def _on_heatmap(self, msg):   self.latest_heatmap_msg   = msg
    def _on_odom(self, msg):      self.latest_odom_msg       = msg

    # ── Update ───────────────────────────────────────────────────────────────

    def on_update(self, dt: float):
        if self.latest_odom_msg is None:
            return
        odom    = self.latest_odom_msg
        robot_x = odom.get('x', 0.0)
        robot_y = odom.get('y', 0.0)
        cell    = (int(robot_y / CELL_SIZE), int(robot_x / CELL_SIZE))
        self.traversed_cells.add(cell)

        has_weed = self._cell_has_weed(cell)
        dets     = self.latest_detection_msg.get('detections', []) if self.latest_detection_msg else []
        heatmap  = self.latest_heatmap_msg.get('heatmap') if self.latest_heatmap_msg else None
        density  = float(heatmap[cell[0], cell[1]]) if (
            heatmap is not None and
            0 <= cell[0] < heatmap.shape[0] and
            0 <= cell[1] < heatmap.shape[1]
        ) else 0.0
        max_conf = max((d.get('confidence', 0) for d in dets), default=0.0)

        # ── UNIFORM: spray every visited cell ──────────────────────────────
        self.trackers[UNIFORM].record(cell, has_weed)

        # ── DETECTION_ONLY: any detection present ─────────────────────────
        if max_conf > self.conf_threshold:
            self.trackers[DETECTION_ONLY].record(cell, has_weed)

        # ── DUAL_THRESHOLD (proposed): H > τ_d AND max_conf > τ_c ────────
        if density > self.density_threshold and max_conf > self.conf_threshold:
            self.trackers[DUAL_THRESHOLD].record(cell, has_weed)
            if not self.spray_active:
                self.spray_active = True
                self.spray_timer  = self.spray_duration
                self.total_chemical += self.spray_duration * self.flow_rate
                if self.robot:
                    self.robot.activate_sprayer(max_conf, density)
        else:
            if self.robot:
                self.robot.deactivate_sprayer()

        # Timer
        if self.spray_active:
            self.spray_timer -= dt
            if self.spray_timer <= 0:
                self.spray_active = False
                if self.robot:
                    self.robot.deactivate_sprayer()

        self.publish(TOPIC_SPRAY_CMD, {
            'active':         self.spray_active,
            'total_events':   self.trackers[DUAL_THRESHOLD].spray_count,
            'total_chemical': self.total_chemical,
            'timestamp':      odom.get('timestamp', 0),
        })

    # ── Direct spray decision (used from SimulationEngine) ────────────────────

    def decide(
        self,
        robot_x: float,
        robot_y: float,
        detections: List[dict],
        heatmap: Optional[np.ndarray],
    ) -> Dict[str, bool]:
        """
        Evaluate all three strategies for a single cell visit.

        Returns dict of {strategy_name: did_spray}.
        """
        cell     = (int(robot_y / CELL_SIZE), int(robot_x / CELL_SIZE))
        has_weed = self._cell_has_weed(cell)
        max_conf = max((d.get('confidence', 0) for d in detections), default=0.0)
        density  = 0.0
        if heatmap is not None and (
                0 <= cell[0] < heatmap.shape[0] and
                0 <= cell[1] < heatmap.shape[1]):
            density = float(heatmap[cell[0], cell[1]])

        self.traversed_cells.add(cell)
        self.trackers[UNIFORM].record(cell, has_weed)

        sprayed = {UNIFORM: True, DETECTION_ONLY: False, DUAL_THRESHOLD: False}

        if max_conf > self.conf_threshold:
            self.trackers[DETECTION_ONLY].record(cell, has_weed)
            sprayed[DETECTION_ONLY] = True

        if density > self.density_threshold and max_conf > self.conf_threshold:
            self.trackers[DUAL_THRESHOLD].record(cell, has_weed)
            sprayed[DUAL_THRESHOLD] = True
            if not self.spray_active:
                self.spray_active    = True
                self.spray_timer     = self.spray_duration
                self.total_chemical += self.spray_duration * self.flow_rate
                if self.robot:
                    self.robot.activate_sprayer(max_conf, density)
        else:
            if self.robot:
                self.robot.deactivate_sprayer()

        return sprayed

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cell_has_weed(self, cell: Tuple[int, int]) -> bool:
        if self.weed_ground_truth is None:
            return False
        r, c = cell
        if 0 <= r < self.weed_ground_truth.shape[0] and \
           0 <= c < self.weed_ground_truth.shape[1]:
            return self.weed_ground_truth[r, c] > 0.0
        return False

    # ── Summary metrics ───────────────────────────────────────────────────────

    def get_all_strategy_metrics(self, total_weed_cells: int) -> dict:
        """
        Return the full strategy comparison dict for simulation_results.json.
        """
        uniform_count = self.trackers[UNIFORM].spray_count
        return {
            UNIFORM:        self.trackers[UNIFORM].metrics(uniform_count, total_weed_cells),
            DETECTION_ONLY: self.trackers[DETECTION_ONLY].metrics(uniform_count, total_weed_cells),
            DUAL_THRESHOLD: self.trackers[DUAL_THRESHOLD].metrics(uniform_count, total_weed_cells),
        }

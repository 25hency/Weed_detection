"""
Module 4 — Adaptive A* Navigation & Path Planner  (AFRS Paper §3.5)

Cost function (paper eq. 7):
    C(n) = α·C_b(n) + β·C_o(n) − γ·H(n)

Named coefficients (paper §3.5):
    ALPHA = 1.0   — base distance weight
    BETA  = 2.0   — obstacle proximity weight
    GAMMA = 1.5   — weed-density reward weight

C_b(n): Euclidean distance in METRES
    = CELL_SIZE × sqrt(dr² + dc²)
    = 0.5 m (cardinal) or ≈ 0.707 m (diagonal)

C_o(n): Obstacle proximity penalty
    = infinity if cell is obstacle
    = BETA × 5.0  if cell is adjacent (distance 1) to an obstacle
    = 0.0 otherwise

Priority zones: cells where H(i,j) > 0.3 (top ~30 % density)

Dynamic replanning trigger: sum of heatmap changes in forward path region
    exceeds delta_threshold = 0.15

Boustrophedon baseline: serpentine row-by-row covering all non-obstacle cells.
Path length in METRES for both adaptive and boustrophedon paths.

Metrics logged:
    path_length_adaptive_m, path_length_boustrophedon_m,
    priority_coverage_ratio, total_coverage_ratio, replanning_events
"""

import numpy as np
import heapq
import time
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.node_base import NodeBase
from simulation.farm_world import CELL_SIZE, GRID_SIZE

logger = logging.getLogger(__name__)

TOPIC_HEATMAP = '/heatmap/grid'
TOPIC_ODOM    = '/odom/pose'
TOPIC_LIDAR   = '/lidar/scan'
TOPIC_PATH    = '/planner/path'

# ── Named cost coefficients (paper §3.5) ─────────────────────────────────────
ALPHA = 1.0   # base distance cost weight
BETA  = 2.0   # obstacle proximity weight
GAMMA = 1.5   # weed-density reward weight

# Priority zone density threshold
PRIORITY_THRESHOLD    = 0.3
REPLAN_DELTA_THRESHOLD = 0.15


@dataclass
class PlannerConfig:
    """Configuration for the Adaptive A* planner."""
    alpha: float = ALPHA
    beta:  float = BETA
    gamma: float = GAMMA
    replan_threshold: float = REPLAN_DELTA_THRESHOLD
    grid_resolution:  float = CELL_SIZE


class AStarNode:
    """Node in the A* search graph."""
    __slots__ = ['row', 'col', 'g', 'h', 'f', 'parent']

    def __init__(self, row: int, col: int, g: float = float('inf'),
                 h: float = 0.0, parent=None):
        self.row    = row
        self.col    = col
        self.g      = g
        self.h      = h
        self.f      = g + h
        self.parent = parent

    def __lt__(self, other): return self.f < other.f
    def __eq__(self, other): return self.row == other.row and self.col == other.col
    def __hash__(self):      return hash((self.row, self.col))


class AdaptiveAStarPlanner(NodeBase):
    """
    Adaptive A* Navigation Planner.

    Uses the modified cost function C(n) = α·C_b(n) + β·C_o(n) − γ·H(n)
    with ALPHA=1.0, BETA=2.0, GAMMA=1.5 (paper §3.5).

    All path lengths reported in METRES.
    """

    def __init__(
        self,
        occupancy_grid: np.ndarray = None,
        config: PlannerConfig = None,
        rate_hz: float = 5.0,
    ):
        super().__init__('astar_planner', rate_hz=rate_hz)

        self.config         = config or PlannerConfig()
        self.occupancy_grid = occupancy_grid
        self.heatmap        = None

        self.grid_h = GRID_SIZE
        self.grid_w = GRID_SIZE
        if occupancy_grid is not None:
            self.grid_h, self.grid_w = occupancy_grid.shape

        # Current planned path
        self.current_path: List[Tuple[int, int]] = []
        self.path_index = 0
        self.goal       = None
        self.start_pos  = None

        # State subscriptions
        self.latest_heatmap_msg = None
        self.latest_odom_msg    = None

        # Performance metrics
        self.planning_times:   List[float]         = []
        self.replan_count:     int                  = 0
        self.visited_cells:    set                  = set()
        self.priority_visited: set                  = set()

        # Path metric accumulators
        self.total_path_length_m:    float = 0.0
        self.first_priority_dist_m:  float = -1.0  # metres before first priority cell
        self._dist_so_far_m:         float = 0.0

        # 8-connected neighbour offsets
        self._neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

    # ── Node lifecycle ────────────────────────────────────────────────────────

    def set_occupancy_grid(self, grid: np.ndarray):
        self.occupancy_grid = grid
        self.grid_h, self.grid_w = grid.shape

    def on_start(self):
        self.subscribe(TOPIC_HEATMAP, self._on_heatmap)
        self.subscribe(TOPIC_ODOM,    self._on_odom)
        logger.info(
            f"A* planner started — α={self.config.alpha}, "
            f"β={self.config.beta}, γ={self.config.gamma}"
        )

    def _on_heatmap(self, msg):
        self.latest_heatmap_msg = msg
        self.heatmap = msg.get('heatmap')
        if msg.get('heatmap_changed', False) and self.current_path:
            delta = msg.get('forward_delta', 0.0)
            if delta > self.config.replan_threshold:
                self._trigger_replan()

    def _on_odom(self, msg):
        self.latest_odom_msg = msg

    def on_update(self, dt: float):
        if self.current_path:
            path_msg = {
                'path':         self.current_path,
                'path_index':   self.path_index,
                'path_length_m': self._compute_path_length(self.current_path),
                'timestamp':    time.time(),
                'replan_count': self.replan_count,
            }
            self.publish(TOPIC_PATH, path_msg)

    def _trigger_replan(self):
        if self.latest_odom_msg and self.goal:
            odom = self.latest_odom_msg
            sr   = int(odom['y'] / self.config.grid_resolution)
            sc   = int(odom['x'] / self.config.grid_resolution)
            start = (
                max(0, min(sr, self.grid_h - 1)),
                max(0, min(sc, self.grid_w - 1)),
            )
            new_path = self.plan_path(start, self.goal)
            if new_path:
                self.current_path = new_path
                self.path_index   = 0
                self.replan_count += 1
                logger.debug(f"Replanned (#{self.replan_count})")

    # ── A* planner ───────────────────────────────────────────────────────────

    def plan_path(
        self,
        start: Tuple[int, int],
        goal:  Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Plan path using C(n) = α·C_b(n) + β·C_o(n) − γ·H(n).

        Returns list of (row, col) waypoints; empty list if no path found.
        """
        if self.occupancy_grid is None:
            logger.error("No occupancy grid set!")
            return []

        t0 = time.perf_counter()
        self.start_pos = start
        self.goal      = goal

        if self._is_blocked(*start):
            start = self._find_nearest_free(start)
        if self._is_blocked(*goal):
            goal = self._find_nearest_free(goal)

        open_set:   list                     = []
        closed_set: set                      = set()
        g_scores:   Dict[Tuple, float]       = {start: 0.0}
        came_from:  Dict[Tuple, Tuple]       = {}

        s_node = AStarNode(start[0], start[1], g=0.0,
                           h=self._heuristic(start, goal))
        s_node.f = s_node.g + s_node.h
        heapq.heappush(open_set, s_node)

        while open_set:
            cur  = heapq.heappop(open_set)
            cpos = (cur.row, cur.col)

            if cpos == goal:
                path = self._reconstruct_path(came_from, cpos)
                self.planning_times.append((time.perf_counter() - t0) * 1000)
                self.current_path = path
                self.path_index   = 0
                return path

            if cpos in closed_set:
                continue
            closed_set.add(cpos)

            for dr, dc in self._neighbors:
                nr, nc = cur.row + dr, cur.col + dc
                npos   = (nr, nc)

                if not (0 <= nr < self.grid_h and 0 <= nc < self.grid_w):
                    continue
                if npos in closed_set:
                    continue
                if self._is_blocked(nr, nc):
                    continue

                # ── Cost function (paper §3.5) ────────────────────────────
                # C_b(n): Euclidean distance in metres
                Cb = CELL_SIZE * np.sqrt(dr * dr + dc * dc)

                # C_o(n): obstacle proximity penalty (paper eq. 8)
                Co = self._obstacle_cost(nr, nc)

                # H(n): weed-density reward
                Hn = self._weed_density_reward(nr, nc)

                step_cost = (
                    self.config.alpha * Cb
                    + self.config.beta  * Co
                    - self.config.gamma * Hn
                )
                step_cost = max(1e-4, step_cost)   # keep positive

                tent_g = cur.g + step_cost
                if tent_g < g_scores.get(npos, float('inf')):
                    g_scores[npos]   = tent_g
                    came_from[npos]  = cpos
                    h = self._heuristic(npos, goal)
                    n_node = AStarNode(nr, nc, g=tent_g, h=h)
                    n_node.f = tent_g + h
                    heapq.heappush(open_set, n_node)

        self.planning_times.append((time.perf_counter() - t0) * 1000)
        logger.warning(f"No path found {start} → {goal}")
        return []

    # ── Cost components ───────────────────────────────────────────────────────

    def _obstacle_cost(self, row: int, col: int) -> float:
        """
        C_o(n) per paper §3.5:
          - inf (blocked cell already excluded in plan_path)
          - BETA × 5.0 if any direct neighbour (distance 1) is an obstacle
          - 0.0 otherwise
        """
        if self.occupancy_grid is None:
            return 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
                    if self.occupancy_grid[nr, nc] > 0.5:
                        return self.config.beta * 5.0
        return 0.0

    def _weed_density_reward(self, row: int, col: int) -> float:
        if self.heatmap is None:
            return 0.0
        if 0 <= row < self.heatmap.shape[0] and 0 <= col < self.heatmap.shape[1]:
            return float(self.heatmap[row, col])
        return 0.0

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Octile distance heuristic — admissible for 8-connected grids."""
        dr = abs(pos[0] - goal[0])
        dc = abs(pos[1] - goal[1])
        return CELL_SIZE * (max(dr, dc) + (np.sqrt(2) - 1) * min(dr, dc))

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _compute_path_length(self, path: List[Tuple[int, int]]) -> float:
        """Sum of inter-cell Euclidean distances in METRES."""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            dr = path[i][0] - path[i - 1][0]
            dc = path[i][1] - path[i - 1][1]
            total += np.sqrt(dr * dr + dc * dc) * CELL_SIZE
        return total

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple, Tuple],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _is_blocked(self, row: int, col: int) -> bool:
        if not (0 <= row < self.grid_h and 0 <= col < self.grid_w):
            return True
        return self.occupancy_grid[row, col] > 0.5

    def _find_nearest_free(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        for radius in range(1, 20):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if (0 <= nr < self.grid_h and 0 <= nc < self.grid_w
                            and not self._is_blocked(nr, nc)):
                        return (nr, nc)
        return pos

    # ── Coverage & priority tracking ──────────────────────────────────────────

    def record_visit(self, row: int, col: int, heatmap: np.ndarray) -> None:
        """
        Record that the robot visited cell (row, col).
        Updates priority-zone and total coverage counters.
        """
        self.visited_cells.add((row, col))
        if (0 <= row < heatmap.shape[0] and 0 <= col < heatmap.shape[1]
                and heatmap[row, col] > PRIORITY_THRESHOLD):
            if (row, col) not in self.priority_visited:
                self.priority_visited.add((row, col))
                if self.first_priority_dist_m < 0:
                    self.first_priority_dist_m = self._dist_so_far_m

    def advance_distance(self, delta_m: float) -> None:
        """Accumulate distance travelled (metres)."""
        self._dist_so_far_m      += delta_m
        self.total_path_length_m += delta_m

    def get_navigation_metrics(
        self,
        boustrophedon_path: List[Tuple[int, int]],
        occupancy_grid: np.ndarray,
        heatmap: np.ndarray,
    ) -> dict:
        """
        Compute and return navigation metrics for simulation_results.json.
        """
        adaptive_len_m     = self.total_path_length_m
        boustrophedon_len_m = self._compute_path_length(boustrophedon_path)

        # Total non-obstacle traversable cells
        non_obstacle = int((occupancy_grid < 0.5).sum())
        total_coverage = (
            len(self.visited_cells) / non_obstacle if non_obstacle > 0 else 0.0
        )

        # Priority cells in the world
        priority_cells_total = int((heatmap > PRIORITY_THRESHOLD).sum())
        priority_coverage = (
            len(self.priority_visited) / priority_cells_total
            if priority_cells_total > 0 else 0.0
        )

        return {
            "path_length_adaptive_m":    round(adaptive_len_m, 3),
            "path_length_boustrophedon_m": round(boustrophedon_len_m, 3),
            "priority_coverage_ratio":   round(min(priority_coverage, 1.0), 4),
            "total_coverage_ratio":      round(min(total_coverage, 1.0), 4),
            "replanning_events":         self.replan_count,
        }

    # ── Boustrophedon baseline ────────────────────────────────────────────────

    @staticmethod
    def generate_boustrophedon_path(
        grid: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Serpentine (lawn-mower) path covering all non-obstacle interior cells.

        Traverses every row left-to-right then right-to-left alternately.
        Returns list of (row, col). Total path length in metres via
        FarmWorld.path_length_meters or _compute_path_length.
        """
        grid_h, grid_w = grid.shape
        path      = []
        left_to_right = True

        for row in range(1, grid_h - 1):
            cols = range(1, grid_w - 1) if left_to_right else range(grid_w - 2, 0, -1)
            for col in cols:
                if grid[row, col] < 0.5:   # not an obstacle
                    path.append((row, col))
            left_to_right = not left_to_right

        return path

    # ── Performance ───────────────────────────────────────────────────────────

    def get_avg_planning_time(self) -> float:
        if not self.planning_times:
            return 0.0
        return float(np.mean(self.planning_times[-10:]))

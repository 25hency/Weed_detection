"""
Farm World Simulator — Virtual Agricultural Environment (AFRS Paper §3.1)

10 × 10 metre field discretised into 0.5 × 0.5 m cells → 20 × 20 grid.

Supports 5 named scenarios:
    LOW_DENSITY    — ~10 % weed cells, no occlusion
    MEDIUM_DENSITY — ~25 % weed cells, sparse overlap
    HIGH_DENSITY   — ~45 % weed cells, clustered groups
    SHADOWED       — same as MEDIUM but camera noise sigma × 2
    OVERLAPPING    — ~40 % weed cells, bounding-box overlap > 40 %

Path length is computed in METRES (CELL_SIZE × Euclidean cell distance).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ── Grid constants (paper §3.1) ──────────────────────────────────────────────
GRID_SIZE  = 20          # cells per axis (20 × 20)
CELL_SIZE  = 0.5         # metres per cell
FIELD_SIZE = GRID_SIZE * CELL_SIZE   # 10.0 m

# ── Scenario names ────────────────────────────────────────────────────────────
LOW_DENSITY    = "LOW_DENSITY"
MEDIUM_DENSITY = "MEDIUM_DENSITY"
HIGH_DENSITY   = "HIGH_DENSITY"
SHADOWED       = "SHADOWED"
OVERLAPPING    = "OVERLAPPING"

SCENARIOS = [LOW_DENSITY, MEDIUM_DENSITY, HIGH_DENSITY, SHADOWED, OVERLAPPING]

# Scenario parameters
_SCENARIO_CFG = {
    LOW_DENSITY:    {"weed_fraction": 0.10, "cluster": False, "noise_sigma": 0.10, "overlap": False},
    MEDIUM_DENSITY: {"weed_fraction": 0.25, "cluster": False, "noise_sigma": 0.10, "overlap": False},
    HIGH_DENSITY:   {"weed_fraction": 0.45, "cluster": True,  "noise_sigma": 0.10, "overlap": False},
    SHADOWED:       {"weed_fraction": 0.25, "cluster": False, "noise_sigma": 0.20, "overlap": False},
    OVERLAPPING:    {"weed_fraction": 0.40, "cluster": True,  "noise_sigma": 0.10, "overlap": True},
}


@dataclass
class WeedPatch:
    """A cluster of weeds in the field."""
    center_row: int
    center_col: int
    radius_cells: float
    density: float
    species_id: int


@dataclass
class Obstacle:
    row: int
    col: int


@dataclass
class FarmWorld:
    """
    Grid-based farm world: 10 × 10 m, 20 × 20 cells, cell = 0.5 m.

    Parameters
    ----------
    scenario : str
        One of LOW_DENSITY, MEDIUM_DENSITY, HIGH_DENSITY, SHADOWED, OVERLAPPING.
    seed : int
        RNG seed for reproducibility.
    """
    scenario: str = MEDIUM_DENSITY
    seed: int = 42

    # Populated in __post_init__
    weed_patches:    List[WeedPatch] = field(default_factory=list)
    obstacles:       List[Obstacle]  = field(default_factory=list)
    occupancy_grid:  Optional[np.ndarray] = None  # 1 = obstacle
    weed_ground_truth: Optional[np.ndarray] = None  # fraction in [0,1]
    camera_noise_sigma: float = 0.10

    # Expose for downstream compatibility
    width:      float = FIELD_SIZE
    height:     float = FIELD_SIZE
    resolution: float = CELL_SIZE
    grid_w:     int   = GRID_SIZE
    grid_h:     int   = GRID_SIZE

    def __post_init__(self):
        if self.scenario not in _SCENARIO_CFG:
            raise ValueError(f"Unknown scenario '{self.scenario}'. Choose from {SCENARIOS}")
        cfg = _SCENARIO_CFG[self.scenario]
        self.camera_noise_sigma = cfg["noise_sigma"]
        self._generate_world(cfg)

    # ── World generation ──────────────────────────────────────────────────────

    def _generate_world(self, cfg: dict) -> None:
        rng = np.random.RandomState(self.seed)
        G = GRID_SIZE

        self.occupancy_grid     = np.zeros((G, G), dtype=np.float32)
        self.weed_ground_truth  = np.zeros((G, G), dtype=np.float32)

        # Border obstacles (always present)
        self.occupancy_grid[0, :]  = 1.0
        self.occupancy_grid[-1, :] = 1.0
        self.occupancy_grid[:, 0]  = 1.0
        self.occupancy_grid[:, -1] = 1.0

        # Interior obstacles (2–4)
        for _ in range(rng.randint(2, 5)):
            r = rng.randint(2, G - 2)
            c = rng.randint(2, G - 2)
            self.occupancy_grid[r, c] = 1.0
            self.obstacles.append(Obstacle(r, c))

        # Weed placement
        interior = (G - 2) * (G - 2)   # 324 cells
        target_cells = int(cfg["weed_fraction"] * interior)

        if cfg["cluster"]:
            self._place_clustered_weeds(rng, target_cells)
        else:
            self._place_random_weeds(rng, target_cells)

        if cfg["overlap"]:
            self._ensure_bbox_overlap(rng)

        actual_weed_cells = int((self.weed_ground_truth > 0).sum())
        logger.info(
            f"FarmWorld [{self.scenario}] — "
            f"weed cells: {actual_weed_cells}/{interior} ({actual_weed_cells/interior*100:.0f}%), "
            f"obstacles: {len(self.obstacles)}, "
            f"noise_sigma={self.camera_noise_sigma}"
        )

    def _place_random_weeds(self, rng: np.random.RandomState, target: int) -> None:
        G = GRID_SIZE
        placed = 0
        attempts = 0
        while placed < target and attempts < target * 10:
            r = rng.randint(1, G - 1)
            c = rng.randint(1, G - 1)
            if self.occupancy_grid[r, c] == 0 and self.weed_ground_truth[r, c] == 0:
                density = rng.uniform(0.3, 0.9)
                species = rng.randint(0, 15)
                self.weed_ground_truth[r, c] = density
                self.weed_patches.append(WeedPatch(r, c, 0.5, density, species))
                placed += 1
            attempts += 1

    def _place_clustered_weeds(self, rng: np.random.RandomState, target: int) -> None:
        """Place weeds in Gaussian clusters until target cell count is reached."""
        G = GRID_SIZE
        placed_cells = set()
        # Use enough clusters to cover the target; each cluster covers ~pi*r^2 cells
        num_clusters = max(4, target // 6)

        for ci in range(num_clusters):
            cr = rng.randint(3, G - 3)
            cc = rng.randint(3, G - 3)
            radius = rng.uniform(1.5, 3.0)
            density_peak = rng.uniform(0.6, 1.0)
            species = rng.randint(0, 15)
            self.weed_patches.append(WeedPatch(cr, cc, radius, density_peak, species))
            for dr in range(-int(radius) - 1, int(radius) + 2):
                for dc in range(-int(radius) - 1, int(radius) + 2):
                    nr, nc = cr + dr, cc + dc
                    if not (1 <= nr < G - 1 and 1 <= nc < G - 1):
                        continue
                    if self.occupancy_grid[nr, nc] > 0:
                        continue
                    dist = np.sqrt(dr ** 2 + dc ** 2)
                    if dist <= radius:
                        sigma = radius / 2.0
                        val = density_peak * np.exp(-dist ** 2 / (2 * sigma ** 2))
                        val = max(val, 0.35)   # minimum weed density so it's detectable
                        self.weed_ground_truth[nr, nc] = max(
                            self.weed_ground_truth[nr, nc], val
                        )
                        placed_cells.add((nr, nc))
            if len(placed_cells) >= target:
                break

        # If clusters produced fewer cells than target, fill with random weeds
        attempts = 0
        while len(placed_cells) < target and attempts < target * 10:
            r = rng.randint(1, G - 1)
            c = rng.randint(1, G - 1)
            if self.occupancy_grid[r, c] == 0 and self.weed_ground_truth[r, c] == 0:
                density = rng.uniform(0.35, 0.85)
                species = rng.randint(0, 15)
                self.weed_ground_truth[r, c] = density
                self.weed_patches.append(WeedPatch(r, c, 0.5, density, species))
                placed_cells.add((r, c))
            attempts += 1

    def _ensure_bbox_overlap(self, rng: np.random.RandomState) -> None:
        """Force patch pairs to be adjacent so projected bboxes overlap > 40 %."""
        G = GRID_SIZE
        num_pairs = max(2, len(self.weed_patches) // 3)
        for _ in range(num_pairs):
            r = rng.randint(2, G - 3)
            c = rng.randint(2, G - 3)
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                for row, col in [(r, c), (nr, nc)]:
                    if self.occupancy_grid[row, col] == 0:
                        d = rng.uniform(0.5, 0.9)
                        self.weed_ground_truth[row, col] = max(
                            self.weed_ground_truth[row, col], d
                        )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """World metres → (row, col) grid indices."""
        col = int(x / CELL_SIZE)
        row = int(y / CELL_SIZE)
        col = max(0, min(col, GRID_SIZE - 1))
        row = max(0, min(row, GRID_SIZE - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Grid cell centre → world metres (x, y)."""
        x = (col + 0.5) * CELL_SIZE
        y = (row + 0.5) * CELL_SIZE
        return x, y

    # ── Query helpers ─────────────────────────────────────────────────────────

    def is_obstacle(self, row: int, col: int) -> bool:
        return bool(self.occupancy_grid[row, col] > 0.5)

    def get_weed_density(self, x: float, y: float) -> float:
        row, col = self.world_to_grid(x, y)
        return float(self.weed_ground_truth[row, col])

    def get_weed_density_at_cell(self, row: int, col: int) -> float:
        return float(self.weed_ground_truth[row, col])

    # ── Simulated sensor outputs ──────────────────────────────────────────────

    def get_camera_detections(
        self,
        robot_x: float,
        robot_y: float,
        fov_width: float = 2.0,
        fov_depth: float = 3.0,
        robot_theta: float = 0.0,
    ) -> List[dict]:
        """
        Simulate weed detections in the robot's camera FOV.
        Noise is scaled by scenario's camera_noise_sigma.
        Returns raw ground-truth-style detections; confidence is assigned by DetectorNode.
        """
        detections = []
        cos_t = np.cos(robot_theta)
        sin_t = np.sin(robot_theta)

        for patch in self.weed_patches:
            wx, wy = self.grid_to_world(patch.center_row, patch.center_col)
            dx = wx - robot_x
            dy = wy - robot_y

            # Robot-frame coordinates
            local_x =  dx * cos_t + dy * sin_t   # forward
            local_y = -dx * sin_t + dy * cos_t   # lateral

            if 0 < local_x < fov_depth and abs(local_y) < fov_width / 2:
                noise_x = np.random.normal(0, self.camera_noise_sigma)
                noise_y = np.random.normal(0, self.camera_noise_sigma)
                bw = min(0.4, patch.radius_cells * CELL_SIZE / fov_width)
                bh = min(0.4, patch.radius_cells * CELL_SIZE / fov_depth)
                cx_norm = 0.5 + local_y / fov_width
                cy_norm = local_x / fov_depth
                detections.append({
                    "class_id":    patch.species_id,
                    "confidence":  patch.density,           # raw; will be recomputed
                    "bbox":        [cx_norm, cy_norm, bw, bh],
                    "world_x":     wx + noise_x,
                    "world_y":     wy + noise_y,
                    "cell_i":      patch.center_row,
                    "cell_j":      patch.center_col,
                    "gt_density":  patch.density,
                })
        return detections

    def get_lidar_scan(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        num_rays: int = 360,
        max_range: float = 5.0,
    ) -> np.ndarray:
        ranges = np.full(num_rays, max_range, dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        step = CELL_SIZE * 0.4
        for i, angle in enumerate(angles):
            abs_angle = robot_theta + angle
            ddx = np.cos(abs_angle) * step
            ddy = np.sin(abs_angle) * step
            x, y, dist = robot_x, robot_y, 0.0
            while dist < max_range:
                x += ddx; y += ddy; dist += step
                if not (0 <= x < FIELD_SIZE and 0 <= y < FIELD_SIZE):
                    ranges[i] = dist; break
                row, col = self.world_to_grid(x, y)
                if self.occupancy_grid[row, col] > 0.5:
                    ranges[i] = dist + np.random.normal(0, 0.02); break
        return ranges

    # ── Path-length helper (metres) ───────────────────────────────────────────

    @staticmethod
    def path_length_meters(path: List[Tuple[int, int]]) -> float:
        """Sum of inter-cell Euclidean distances in metres."""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            dr = path[i][0] - path[i - 1][0]
            dc = path[i][1] - path[i - 1][1]
            total += np.sqrt(dr * dr + dc * dc) * CELL_SIZE
        return total

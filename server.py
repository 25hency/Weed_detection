"""
AFRS Backend API Server — Flask bridge between Python backend and HTML frontend.

Connects the visual simulator (HTML/JS) to the real AFRS simulation engine
so the frontend can request real-time backend-computed results instead of
using its own duplicate JavaScript algorithms.

Usage:
    python server.py

Opens at http://localhost:5000
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from simulation.farm_world import (
    FarmWorld, CELL_SIZE, GRID_SIZE,
    LOW_DENSITY, MEDIUM_DENSITY, HIGH_DENSITY, SHADOWED, OVERLAPPING, SCENARIOS
)
from simulation.robot import DifferentialDriveRobot, ROBOT_SPEED, TRAVERSAL_TIME
from heatmap.heatmap_node import HeatmapNode, DECAY_FACTOR
from navigation.astar_planner import (
    AdaptiveAStarPlanner, PlannerConfig,
    ALPHA, BETA, GAMMA, PRIORITY_THRESHOLD
)
from spraying.sprayer_node import (
    SprayerNode, TAU_DENSITY, TAU_CONFIDENCE,
    UNIFORM, DETECTION_ONLY, DUAL_THRESHOLD
)

# Import ScenarioEngine from main.py
from main import ScenarioEngine, NUM_CYCLES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('server')

# ══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(
    __name__,
    static_folder=str(PROJECT_ROOT / 'visual_simulation'),
    static_url_path=''
)
CORS(app)

# ── In-memory cache for completed scenario runs ──────────────────────────────
_scenario_cache: Dict[str, dict] = {}

# ── Scenario name mapping (frontend dropdown → Python scenario key) ──────────
SCENARIO_KEY_MAP = {
    'low':         LOW_DENSITY,
    'medium':      MEDIUM_DENSITY,
    'high':        HIGH_DENSITY,
    'shadowed':    SHADOWED,
    'overlapping': OVERLAPPING,
}

# Reverse map for API responses
SCENARIO_LABEL_MAP = {
    LOW_DENSITY:    'Low Density',
    MEDIUM_DENSITY: 'Medium Density',
    HIGH_DENSITY:   'High Density',
    SHADOWED:       'Shadowed',
    OVERLAPPING:    'Overlapping',
}


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Static files (serve the frontend)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def serve_index():
    """Serve the frontend index.html."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve any file from the visual_simulation directory."""
    return send_from_directory(app.static_folder, path)


# ══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for frontend auto-detection."""
    return jsonify({
        'status': 'ok',
        'engine': 'AFRS Simulation Engine',
        'version': '1.0',
        'scenarios': list(SCENARIOS),
        'grid_size': GRID_SIZE,
        'cell_size_m': CELL_SIZE,
    })


@app.route('/api/scenarios', methods=['GET'])
def list_scenarios():
    """List available scenarios and which ones are cached."""
    scenarios = []
    for key, label in SCENARIO_LABEL_MAP.items():
        scenarios.append({
            'key': key,
            'label': label,
            'cached': key in _scenario_cache,
        })
    return jsonify({'scenarios': scenarios})


@app.route('/api/run', methods=['POST'])
def run_scenario():
    """
    Run the full AFRS simulation for a given scenario.

    Request JSON:
        {
            "scenario": "low" | "medium" | "high" | "shadowed" | "overlapping",
            "num_cycles": 200  (optional, default: reduced for speed)
        }

    Returns the same JSON structure as visual_data.json per-scenario,
    so the frontend's existing loadRealScenarioData() works unchanged.
    """
    data = request.get_json(force=True, silent=True) or {}
    scenario_input = data.get('scenario', 'medium')

    # Map frontend key to Python scenario name
    scenario = SCENARIO_KEY_MAP.get(scenario_input, scenario_input)
    if scenario not in SCENARIOS:
        return jsonify({'error': f'Unknown scenario: {scenario_input}'}), 400

    # Check cache
    cache_key = scenario
    if cache_key in _scenario_cache:
        logger.info(f"Serving cached result for [{scenario}]")
        return jsonify(_scenario_cache[cache_key])

    # Run simulation (reduced cycles for API responsiveness)
    num_cycles = data.get('num_cycles', 400)
    num_cycles = max(50, min(num_cycles, 2000))

    logger.info(f"Running simulation for [{scenario}] with {num_cycles} cycles...")
    t0 = time.time()

    try:
        eng = ScenarioEngine(scenario=scenario, num_cycles=num_cycles)
        res = eng.run()
    except Exception as e:
        logger.error(f"Simulation failed for [{scenario}]: {e}")
        return jsonify({'error': f'Simulation failed: {str(e)}'}), 500

    elapsed = time.time() - t0
    logger.info(f"Simulation [{scenario}] completed in {elapsed:.1f}s, "
                f"{len(eng.visual_frames)} visual frames")

    # Build response in the same format as visual_data.json per-scenario
    path = list(eng.boustrophedon_path)
    boust_path = [{'row': int(r), 'col': int(c)} for r, c in path]

    obstacles = [[int(eng.farm.occupancy_grid[r][c] > 0.5)
                  for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]

    ground_truth = [[round(float(eng.farm.weed_ground_truth[r][c]), 4)
                     for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]

    result = {
        'scenario': scenario,
        'label': SCENARIO_LABEL_MAP.get(scenario, scenario),
        'num_frames': len(eng.visual_frames),
        'boustrophedon_path': boust_path,
        'obstacles': obstacles,
        'ground_truth': ground_truth,
        'frames': eng.visual_frames,
        'final_metrics': {
            'detection': res['detection'],
            'navigation': res['navigation'],
            'latency_ms': res['latency_ms'],
            'chemical_reduction_pct': round(
                100.0 - res['spraying'][DUAL_THRESHOLD]['chemical_pct'], 2
            ),
        },
        'computation_time_s': round(elapsed, 2),
    }

    # Cache result
    _scenario_cache[cache_key] = result
    logger.info(f"Cached result for [{scenario}]")

    return jsonify(result)


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the scenario cache to force re-computation."""
    _scenario_cache.clear()
    logger.info("Cache cleared")
    return jsonify({'status': 'cleared'})


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  AFRS Visual Simulation — Backend API Server")
    print("=" * 60)
    print(f"  Frontend:  http://localhost:8080")
    print(f"  API Base:  http://localhost:8080/api")
    print(f"  Health:    http://localhost:8080/api/health")
    print(f"  Scenarios: {', '.join(SCENARIOS)}")
    print("=" * 60 + "\n")

    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True,
        threaded=True,
    )

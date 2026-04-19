/* ═══════════════════════════════════════════════════════════════════════════
   AFRS Visual Simulation — Core Engine
   All-in-browser: detection seeding, heatmap, A*, spray decisions, animation
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
    'use strict';

    // ── Constants ─────────────────────────────────────────────────────────────
    const GRID = 20;
    const CELL_SIZE_M = 0.5;   // metres per cell
    const TAU_D = 0.3;         // default density threshold
    const TAU_C = 0.5;         // default confidence threshold
    const NUM_CLASSES = 15;

    // Fixed color palette: 15 visually distinct colors, one per class ID (0–14)
    const CLASS_COLORS = [
        '#EF4444',   // 0  red
        '#F97316',   // 1  orange
        '#EAB308',   // 2  yellow
        '#84CC16',   // 3  lime
        '#06B6D4',   // 4  cyan
        '#38BDF8',   // 5  sky blue
        '#A855F7',   // 6  purple
        '#EC4899',   // 7  pink
        '#F87171',   // 8  coral
        '#14B8A6',   // 9  teal
        '#F59E0B',   // 10 amber
        '#8B5CF6',   // 11 violet
        '#10B981',   // 12 emerald
        '#FB7185',   // 13 rose
        '#6366F1',   // 14 indigo
    ];

    // Persistent cluster → classID map (stable across frames within a run)
    let clusterCenters = [];      // [{r, c, classId}]
    let cellClassIdCache = {};    // "row,col" → classId  (stable once assigned)

    // Scenario configs: weedCount, clusterCount, clusterRadius, noiseMultiplier
    const SCENARIO_CFG = {
        low:         { weedMin: 20, weedMax: 30, clusters: 2, radius: 1.5, noise: 1.0,  label: 'Low Density' },
        medium:      { weedMin: 35, weedMax: 45, clusters: 4, radius: 2.0, noise: 1.0,  label: 'Medium Density' },
        high:        { weedMin: 50, weedMax: 65, clusters: 6, radius: 2.5, noise: 1.0,  label: 'High Density' },
        shadowed:    { weedMin: 30, weedMax: 40, clusters: 3, radius: 2.0, noise: 2.0,  label: 'Shadowed' },
        overlapping: { weedMin: 40, weedMax: 55, clusters: 5, radius: 3.0, noise: 1.0,  label: 'Overlapping' },
    };

    // ── Seeded RNG (Mulberry32) ──────────────────────────────────────────────
    function mulberry32(a) {
        return function() {
            let t = a += 0x6D2B79F5;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    // ── State ────────────────────────────────────────────────────────────────
    let rng = mulberry32(42);
    let canvas, ctx;
    let bgImage = null;
    let imgW = 0, imgH = 0;
    let cellW = 0, cellH = 0;

    // Simulation data
    let detections = [];       // [{row, col, classId, confidence, x, y, w, h}]
    let heatmap = [];          // 20x20 flat array
    let astarPath = [];        // [{row, col}]
    let boustrophedonPath = [];
    let sprayDecisions = [];   // 20x20 flat: 'spray' | 'skip' | null
    let adjacencyColorMap = new Map(); // "row,col" → classId for adjacency checks
    let obstacles = [];        // 20x20 flat: bool

    // Animation state
    let currentFrame = 0;
    let totalFrames = 0;
    let isPlaying = false;
    let animTimer = null;
    let robotPos = { x: 0, y: 0, angle: 0 };
    let sprayFlashes = [];     // [{cx, cy, frame, maxFrames}]
    let visitedCells = new Set();
    let detectedAtFrame = [];  // per-frame detection reveal list
    let heatmapAtFrame = [];   // per-frame heatmap snapshots

    // ── Real data mode ───────────────────────────────────────────────────────
    let realData = null;           // Parsed visual_data.json
    let realScenarioData = null;   // Active scenario's data from realData
    let useRealData = false;       // true when JSON is loaded & active

    // Scenario name mapping: dropdown value -> Python scenario key
    const SCENARIO_KEY_MAP = {
        'low':         'LOW_DENSITY',
        'medium':      'MEDIUM_DENSITY',
        'high':        'HIGH_DENSITY',
        'shadowed':    'SHADOWED',
        'overlapping': 'OVERLAPPING',
    };

    // Layer visibility
    let layers = {
        detection: true,
        heatmap: true,
        path: true,
        spray: true,
        robot: true,
    };

    // Thresholds (user adjustable)
    let tauD = TAU_D;
    let tauC = TAU_C;

    // Speed
    let fps = 15;

    // Metrics accumulators
    let metrics = {
        weedsDetected: 0,
        cellsSprayed: 0,
        cellsSkipped: 0,
        chemicalSaved: 0,
        coverage: 0,
        pathLength: 0,
        latency: 0,
        replans: 0,
    };

    // ── DOM refs ──────────────────────────────────────────────────────────────
    const $ = (id) => document.getElementById(id);

    function init() {
        canvas = $('sim-canvas');
        ctx = canvas.getContext('2d');

        // Upload image
        $('upload-btn').addEventListener('click', () => $('file-input').click());
        $('file-input').addEventListener('change', handleFileUpload);

        // Load simulation JSON
        $('load-json-btn').addEventListener('click', () => $('json-input').click());
        $('json-input').addEventListener('change', handleJsonUpload);

        // Drag & drop on the canvas container
        const container = $('canvas-container');
        container.addEventListener('dragover', (e) => { e.preventDefault(); container.style.outline = '2px dashed var(--accent-green)'; });
        container.addEventListener('dragleave', () => { container.style.outline = 'none'; });
        container.addEventListener('drop', (e) => {
            e.preventDefault();
            container.style.outline = 'none';
            if (e.dataTransfer.files.length) {
                const file = e.dataTransfer.files[0];
                if (file.name.endsWith('.json')) handleJsonFile(file);
                else loadImage(file);
            }
        });

        // Playback
        $('btn-play').addEventListener('click', togglePlay);
        $('btn-reset').addEventListener('click', resetSimulation);
        $('btn-step').addEventListener('click', stepForward);

        // Speed
        $('speed-slider').addEventListener('input', (e) => {
            fps = parseInt(e.target.value);
            $('speed-value').textContent = fps + ' fps';
            if (isPlaying) { clearInterval(animTimer); animTimer = setInterval(stepForward, 1000 / fps); }
        });

        // Scenario
        $('scenario-select').addEventListener('change', () => {
            if (bgImage) regenerateAndReset();
        });

        // Layer toggles
        ['detection', 'heatmap', 'path', 'spray', 'robot'].forEach(id => {
            $('layer-' + id).addEventListener('change', (e) => {
                layers[id] = e.target.checked;
                draw();
            });
        });

        // Threshold sliders
        $('threshold-density').addEventListener('input', (e) => {
            tauD = parseFloat(e.target.value);
            $('val-density').textContent = tauD.toFixed(2);
            if (bgImage) { recomputeSprayDecisions(); draw(); }
        });
        $('threshold-confidence').addEventListener('input', (e) => {
            tauC = parseFloat(e.target.value);
            $('val-confidence').textContent = tauC.toFixed(2);
            if (bgImage) { recomputeSprayDecisions(); draw(); }
        });
    }

    // ── JSON Data Loading ──────────────────────────────────────────────────────
    function handleJsonUpload(e) {
        if (e.target.files.length) handleJsonFile(e.target.files[0]);
    }

    function handleJsonFile(file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
            try {
                realData = JSON.parse(ev.target.result);
                useRealData = true;

                // Show status badge
                const status = $('json-status');
                const scenarioCount = Object.keys(realData.scenarios || {}).length;
                status.textContent = `✓ ${scenarioCount} scenarios loaded`;
                status.className = 'json-status loaded';

                // If image is already loaded, regenerate with real data
                if (bgImage) regenerateAndReset();
            } catch (err) {
                alert('Invalid JSON file: ' + err.message);
            }
        };
        reader.readAsText(file);
    }

    function getActiveRealScenario() {
        if (!realData || !realData.scenarios) return null;
        const dropdownVal = $('scenario-select').value;
        const pyKey = SCENARIO_KEY_MAP[dropdownVal];
        return realData.scenarios[pyKey] || null;
    }

    // ── File Handling ─────────────────────────────────────────────────────────
    function handleFileUpload(e) {
        if (e.target.files.length) loadImage(e.target.files[0]);
    }

    function loadImage(file) {
        if (!file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const img = new Image();
            img.onload = () => {
                bgImage = img;

                // Size canvas to fit container while maintaining aspect ratio
                const container = $('canvas-container');
                const maxW = container.clientWidth - 20;
                const maxH = container.clientHeight - 20;
                const scale = Math.min(maxW / img.width, maxH / img.height, 1);
                imgW = Math.floor(img.width * scale);
                imgH = Math.floor(img.height * scale);
                canvas.width = imgW;
                canvas.height = imgH;
                cellW = imgW / GRID;
                cellH = imgH / GRID;

                // Show canvas, hide placeholder
                canvas.classList.add('active');
                $('placeholder-msg').style.display = 'none';

                regenerateAndReset();
            };
            img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
    }

    // ── Simulation Generation ─────────────────────────────────────────────────
    function regenerateAndReset() {
        // Check if we should use real data for this scenario
        realScenarioData = useRealData ? getActiveRealScenario() : null;

        rng = mulberry32(Date.now() & 0xFFFFFFFF);
        const scenario = $('scenario-select').value;
        const cfg = SCENARIO_CFG[scenario];

        // Reset state
        heatmap = new Float32Array(GRID * GRID);
        sprayDecisions = new Array(GRID * GRID).fill(null);
        obstacles = new Uint8Array(GRID * GRID);
        detections = [];
        visitedCells = new Set();
        clusterCenters = [];
        cellClassIdCache = {};
        adjacencyColorMap = new Map();
        sprayFlashes = [];

        if (realScenarioData && realScenarioData.frames && realScenarioData.frames.length > 0) {
            // ─── REAL DATA MODE ──────────────────────────────────────────
            loadRealScenarioData();
        } else {
            // ─── PROCEDURAL MODE (fallback) ──────────────────────────────
            generateObstacles();
            generateDetections(cfg);
            computeHeatmap();
            generateBoustrophedonPath();
            generateAStarPath();
            recomputeSprayDecisions();
            precomputeFrames();
        }

        // Enable controls
        $('btn-play').disabled = false;
        $('btn-reset').disabled = false;
        $('btn-step').disabled = false;

        currentFrame = 0;
        isPlaying = false;
        updatePlayIcon();
        draw();
        updateMetricsForFrame(0);
    }

    // ── Load Real Scenario Data ────────────────────────────────────────────────
    function loadRealScenarioData() {
        const sd = realScenarioData;
        const frames = sd.frames;
        totalFrames = frames.length;

        // Load obstacles from real data
        if (sd.obstacles) {
            for (let r = 0; r < GRID; r++)
                for (let c = 0; c < GRID; c++)
                    obstacles[r * GRID + c] = sd.obstacles[r][c] ? 1 : 0;
        } else {
            generateObstacles();
        }

        // Load boustrophedon path from real data
        if (sd.boustrophedon_path) {
            boustrophedonPath = sd.boustrophedon_path.map(p => ({ row: p.row, col: p.col }));
        } else {
            generateBoustrophedonPath();
        }

        // Build A* path from real robot positions (one per frame)
        astarPath = frames.map(f => ({ row: f.cell_row, col: f.cell_col }));

        // Collect ALL detections across all frames (cumulative)
        const allDets = new Map(); // key "row,col,classId" -> detection
        for (const frame of frames) {
            for (const d of frame.detections) {
                const key = `${d.cell_i},${d.cell_j},${d.class_id}`;
                if (!allDets.has(key) || d.confidence > allDets.get(key).confidence) {
                    allDets.set(key, d);
                }
            }
        }
        detections = Array.from(allDets.values()).map(d => ({
            row: d.cell_i,
            col: d.cell_j,
            classId: d.class_id,
            confidence: d.confidence,
        }));

        // Pre-compute per-frame data arrays from real frames
        detectedAtFrame = new Array(totalFrames);
        heatmapAtFrame = new Array(totalFrames);

        const cumulativeDets = [];
        const seenDetKeys = new Set();

        for (let f = 0; f < totalFrames; f++) {
            const frame = frames[f];

            // Cumulative detections up to this frame
            for (const d of frame.detections) {
                const key = `${d.cell_i},${d.cell_j},${d.class_id}`;
                if (!seenDetKeys.has(key)) {
                    seenDetKeys.add(key);
                    cumulativeDets.push({
                        row: d.cell_i,
                        col: d.cell_j,
                        classId: d.class_id,
                        confidence: d.confidence,
                    });
                }
            }
            detectedAtFrame[f] = [...cumulativeDets];

            // Heatmap from real data (20x20 grid)
            const hm = new Float32Array(GRID * GRID);
            if (frame.heatmap_grid) {
                for (let r = 0; r < GRID; r++)
                    for (let c = 0; c < GRID; c++)
                        hm[r * GRID + c] = frame.heatmap_grid[r][c];
            }
            heatmapAtFrame[f] = hm;
        }

        // Final heatmap = last frame's heatmap
        const lastHm = heatmapAtFrame[totalFrames - 1];
        heatmap = new Float32Array(lastHm);

        // Spray decisions from real data
        sprayDecisions = new Array(GRID * GRID).fill('skip');
        for (const frame of frames) {
            if (frame.spray_decision) {
                const idx = frame.cell_row * GRID + frame.cell_col;
                sprayDecisions[idx] = 'spray';
            }
        }

        $('frame-counter').textContent = `Frame 0 / ${totalFrames}`;
    }

    function generateObstacles() {
        // Border obstacles
        for (let i = 0; i < GRID; i++) {
            obstacles[0 * GRID + i] = 1;
            obstacles[(GRID - 1) * GRID + i] = 1;
            obstacles[i * GRID + 0] = 1;
            obstacles[i * GRID + (GRID - 1)] = 1;
        }
        // 2–4 interior obstacles
        const numObs = 2 + Math.floor(rng() * 3);
        for (let k = 0; k < numObs; k++) {
            const r = 2 + Math.floor(rng() * (GRID - 4));
            const c = 2 + Math.floor(rng() * (GRID - 4));
            obstacles[r * GRID + c] = 1;
        }
    }

    function generateDetections(cfg) {
        detections = [];
        const count = cfg.weedMin + Math.floor(rng() * (cfg.weedMax - cfg.weedMin + 1));

        if (cfg.clusters > 0) {
            // ── Step 1: Generate cluster center points ────────────────────
            // LOW=2, MEDIUM=4, HIGH=6 (from cfg.clusters)
            clusterCenters = [];
            for (let i = 0; i < cfg.clusters; i++) {
                clusterCenters.push({
                    r: 2 + Math.floor(rng() * (GRID - 4)),
                    c: 2 + Math.floor(rng() * (GRID - 4)),
                    classId: i % 6,   // Assign class IDs 0–5 to cluster centers
                });
            }

            // Pool of remaining class IDs for detections far from all clusters
            const farClassIds = [6, 7, 8, 9, 10, 11, 12, 13, 14];
            let farIdxCounter = 0;

            // ── Step 2: Generate detections around clusters ──────────────
            for (let k = 0; k < count; k++) {
                const center = clusterCenters[Math.floor(rng() * clusterCenters.length)];
                let r, c;
                let attempts = 0;
                do {
                    const dr = gaussianRng() * cfg.radius;
                    const dc = gaussianRng() * cfg.radius;
                    r = Math.round(center.r + dr);
                    c = Math.round(center.c + dc);
                    attempts++;
                } while ((r < 1 || r >= GRID - 1 || c < 1 || c >= GRID - 1 || obstacles[r * GRID + c]) && attempts < 50);

                if (attempts >= 50) continue;

                // ── Step 3: Assign class ID based on spatial cluster ─────
                const cellKey = `${r},${c}`;
                let classId;

                // Check cache first — once assigned, never changes
                if (cellClassIdCache[cellKey] !== undefined) {
                    classId = cellClassIdCache[cellKey];
                } else {
                    // Find nearest cluster center within radius 3
                    let nearestCluster = null;
                    let nearestDist = Infinity;
                    for (const cc of clusterCenters) {
                        const dist = Math.sqrt((r - cc.r) ** 2 + (c - cc.c) ** 2);
                        if (dist <= 3 && dist < nearestDist) {
                            nearestDist = dist;
                            nearestCluster = cc;
                        }
                    }

                    if (nearestCluster) {
                        classId = nearestCluster.classId;
                    } else {
                        // Far from all clusters → assign from remaining IDs 6–14
                        classId = farClassIds[farIdxCounter % farClassIds.length];
                        farIdxCounter++;
                    }

                    cellClassIdCache[cellKey] = classId;
                }

                let conf = 0.5 + rng() * 0.5;
                if (cfg.noise > 1) conf *= (0.7 + rng() * 0.3); // Shadowed: reduce confidence
                conf = Math.min(1.0, Math.max(0.35, conf));

                const bw = 0.3 + rng() * 0.4;
                const bh = 0.3 + rng() * 0.4;
                detections.push({
                    row: r, col: c,
                    classId,
                    confidence: conf,
                    x: c * cellW + (0.5 - bw / 2) * cellW + rng() * cellW * 0.2,
                    y: r * cellH + (0.5 - bh / 2) * cellH + rng() * cellH * 0.2,
                    w: bw * cellW,
                    h: bh * cellH,
                });

                // Track for adjacency checks
                adjacencyColorMap.set(cellKey, classId);
            }

            // ── Step 4: Fix adjacent same-color collisions ───────────────
            resolveAdjacentColorCollisions();
        }
    }

    /**
     * Ensure no two adjacent/touching detection boxes share the same color.
     * Swap to the nearest unused class ID if a collision is found.
     */
    function resolveAdjacentColorCollisions() {
        const dirs = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];

        for (const det of detections) {
            const myId = det.classId;
            const neighbors = [];
            for (const [dr, dc] of dirs) {
                const nk = `${det.row + dr},${det.col + dc}`;
                if (adjacencyColorMap.has(nk)) {
                    neighbors.push(adjacencyColorMap.get(nk));
                }
            }

            // Check if any neighbor shares our classId
            if (neighbors.includes(myId)) {
                // Find first unused classId not in neighbor set
                const neighborSet = new Set(neighbors);
                for (let cid = 0; cid < NUM_CLASSES; cid++) {
                    if (cid !== myId && !neighborSet.has(cid)) {
                        det.classId = cid;
                        const key = `${det.row},${det.col}`;
                        cellClassIdCache[key] = cid;
                        adjacencyColorMap.set(key, cid);
                        break;
                    }
                }
            }
        }
    }

    function gaussianRng() {
        // Box-Muller
        let u = 0, v = 0;
        while (u === 0) u = rng();
        while (v === 0) v = rng();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    // ── Heatmap with Gaussian Spread ──────────────────────────────────────────
    function computeHeatmap() {
        heatmap = new Float32Array(GRID * GRID);
        const sigma = 1.5;

        for (const det of detections) {
            if (det.confidence < 0.5) continue;
            const spread = 3;
            for (let dr = -spread; dr <= spread; dr++) {
                for (let dc = -spread; dc <= spread; dc++) {
                    const nr = det.row + dr;
                    const nc = det.col + dc;
                    if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) continue;
                    const dist2 = dr * dr + dc * dc;
                    const weight = det.confidence * Math.exp(-dist2 / (2 * sigma * sigma));
                    heatmap[nr * GRID + nc] += weight;
                }
            }
        }

        // Normalize to [0, 1]
        let maxVal = 0;
        for (let i = 0; i < heatmap.length; i++) {
            if (heatmap[i] > maxVal) maxVal = heatmap[i];
        }
        if (maxVal > 0) {
            for (let i = 0; i < heatmap.length; i++) {
                heatmap[i] /= maxVal;
            }
        }
    }

    // ── Boustrophedon (Serpentine) Path ────────────────────────────────────────
    function generateBoustrophedonPath() {
        boustrophedonPath = [];
        let leftToRight = true;
        for (let r = 1; r < GRID - 1; r++) {
            if (leftToRight) {
                for (let c = 1; c < GRID - 1; c++) {
                    if (!obstacles[r * GRID + c]) boustrophedonPath.push({ row: r, col: c });
                }
            } else {
                for (let c = GRID - 2; c >= 1; c--) {
                    if (!obstacles[r * GRID + c]) boustrophedonPath.push({ row: r, col: c });
                }
            }
            leftToRight = !leftToRight;
        }
    }

    // ── A* Pathfinding (prioritizes high heatmap cells, no revisits) ─────────
    function generateAStarPath() {
        // Collect all traversable interior cells with heatmap values
        const allCells = [];
        const highCells = [];  // heatmap > 0.3
        for (let r = 1; r < GRID - 1; r++) {
            for (let c = 1; c < GRID - 1; c++) {
                if (!obstacles[r * GRID + c]) {
                    const val = heatmap[r * GRID + c];
                    allCells.push({ row: r, col: c, val });
                    if (val > 0.3) highCells.push({ row: r, col: c, val });
                }
            }
        }
        // Sort high-density cells descending
        highCells.sort((a, b) => b.val - a.val);

        astarPath = [];
        const globalVisited = new Set();
        let current = { row: 1, col: 1 };
        astarPath.push(current);
        globalVisited.add(`${current.row},${current.col}`);

        // Phase 1: Visit high-density clusters first via nearest-neighbor greedy
        const remaining = [...highCells].filter(c => !globalVisited.has(`${c.row},${c.col}`));
        while (remaining.length > 0) {
            // Find nearest unvisited high-density cell
            let bestIdx = 0;
            let bestDist = Infinity;
            for (let i = 0; i < remaining.length; i++) {
                const d = Math.abs(remaining[i].row - current.row) + Math.abs(remaining[i].col - current.col);
                // Bias toward high-value cells: lower effective distance for higher values
                const effectiveDist = d - remaining[i].val * 3;
                if (effectiveDist < bestDist) {
                    bestDist = effectiveDist;
                    bestIdx = i;
                }
            }
            const target = remaining.splice(bestIdx, 1)[0];
            const key = `${target.row},${target.col}`;
            if (globalVisited.has(key)) continue;

            // A* segment to this target
            const seg = astarSegment(current, target);
            for (let i = 1; i < seg.length; i++) {
                const sk = `${seg[i].row},${seg[i].col}`;
                if (!globalVisited.has(sk)) {
                    astarPath.push(seg[i]);
                    globalVisited.add(sk);
                }
            }
            current = target;
        }

        // Phase 2: Sweep remaining unvisited cells in boustrophedon order
        for (const cell of boustrophedonPath) {
            const key = `${cell.row},${cell.col}`;
            if (globalVisited.has(key)) continue;

            const seg = astarSegment(current, cell);
            for (let i = 1; i < seg.length; i++) {
                const sk = `${seg[i].row},${seg[i].col}`;
                if (!globalVisited.has(sk)) {
                    astarPath.push(seg[i]);
                    globalVisited.add(sk);
                }
            }
            current = cell;
        }
    }

    function astarSegment(start, goal) {
        const key = (r, c) => r * GRID + c;
        const open = new MinHeap();
        const gScore = new Map();
        const cameFrom = new Map();
        const startK = key(start.row, start.col);
        const goalK = key(goal.row, goal.col);

        gScore.set(startK, 0);
        open.push({ row: start.row, col: start.col, f: heuristic(start, goal) });

        const dirs = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
        const closed = new Set();

        while (open.size() > 0) {
            const cur = open.pop();
            const ck = key(cur.row, cur.col);

            if (ck === goalK) {
                // Reconstruct
                const path = [{ row: goal.row, col: goal.col }];
                let k = goalK;
                while (cameFrom.has(k)) {
                    k = cameFrom.get(k);
                    path.push({ row: Math.floor(k / GRID), col: k % GRID });
                }
                path.reverse();
                return path;
            }

            if (closed.has(ck)) continue;
            closed.add(ck);

            for (const [dr, dc] of dirs) {
                const nr = cur.row + dr;
                const nc = cur.col + dc;
                if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) continue;
                const nk = key(nr, nc);
                if (closed.has(nk) || obstacles[nk]) continue;

                const dist = Math.sqrt(dr * dr + dc * dc) * CELL_SIZE_M;
                // Cost = distance - γ * heatmap (prefer high-density cells)
                const hVal = heatmap[nk] || 0;
                const stepCost = Math.max(0.01, dist - 1.5 * hVal * CELL_SIZE_M);
                const tentG = (gScore.get(ck) || 0) + stepCost;

                if (tentG < (gScore.get(nk) || Infinity)) {
                    gScore.set(nk, tentG);
                    cameFrom.set(nk, ck);
                    const h = heuristic({ row: nr, col: nc }, goal);
                    open.push({ row: nr, col: nc, f: tentG + h });
                }
            }
        }
        return [start]; // no path
    }

    function heuristic(a, b) {
        const dr = Math.abs(a.row - b.row);
        const dc = Math.abs(a.col - b.col);
        return CELL_SIZE_M * (Math.max(dr, dc) + (Math.SQRT2 - 1) * Math.min(dr, dc));
    }

    // ── Min-Heap ──────────────────────────────────────────────────────────────
    class MinHeap {
        constructor() { this.data = []; }
        size() { return this.data.length; }
        push(val) {
            this.data.push(val);
            this._bubbleUp(this.data.length - 1);
        }
        pop() {
            const top = this.data[0];
            const last = this.data.pop();
            if (this.data.length > 0) {
                this.data[0] = last;
                this._sinkDown(0);
            }
            return top;
        }
        _bubbleUp(i) {
            while (i > 0) {
                const p = (i - 1) >> 1;
                if (this.data[p].f <= this.data[i].f) break;
                [this.data[p], this.data[i]] = [this.data[i], this.data[p]];
                i = p;
            }
        }
        _sinkDown(i) {
            const n = this.data.length;
            while (true) {
                let smallest = i;
                const l = 2 * i + 1, r = 2 * i + 2;
                if (l < n && this.data[l].f < this.data[smallest].f) smallest = l;
                if (r < n && this.data[r].f < this.data[smallest].f) smallest = r;
                if (smallest === i) break;
                [this.data[smallest], this.data[i]] = [this.data[i], this.data[smallest]];
                i = smallest;
            }
        }
    }

    // ── Spray Decisions ───────────────────────────────────────────────────────
    function recomputeSprayDecisions() {
        sprayDecisions = new Array(GRID * GRID).fill(null);

        for (let r = 1; r < GRID - 1; r++) {
            for (let c = 1; c < GRID - 1; c++) {
                const idx = r * GRID + c;
                if (obstacles[idx]) continue;

                const density = heatmap[idx];
                // Check if any detection in this cell exceeds confidence threshold
                const hasConfident = detections.some(d => d.row === r && d.col === c && d.confidence > tauC);

                if (density > tauD && hasConfident) {
                    sprayDecisions[idx] = 'spray';
                } else {
                    sprayDecisions[idx] = 'skip';
                }
            }
        }
    }

    // ── Pre-compute per-frame data ────────────────────────────────────────────
    function precomputeFrames() {
        totalFrames = astarPath.length;
        detectedAtFrame = new Array(totalFrames);
        heatmapAtFrame = new Array(totalFrames);

        // Progressively reveal detections: assign each detection to the frame
        // when the robot first reaches that cell
        const cellToFrameMap = new Map();
        for (let f = 0; f < astarPath.length; f++) {
            const p = astarPath[f];
            const key = `${p.row},${p.col}`;
            if (!cellToFrameMap.has(key)) cellToFrameMap.set(key, f);
            // Also include neighbors (robot scans nearby)
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    const nk = `${p.row + dr},${p.col + dc}`;
                    if (!cellToFrameMap.has(nk)) cellToFrameMap.set(nk, f);
                }
            }
        }

        // Map detections to frames
        const frameDetections = new Array(totalFrames).fill(null).map(() => []);
        for (const det of detections) {
            const key = `${det.row},${det.col}`;
            const frame = cellToFrameMap.get(key) || 0;
            frameDetections[Math.min(frame, totalFrames - 1)].push(det);
        }

        // Cumulative detected list per frame
        let cumulative = [];
        for (let f = 0; f < totalFrames; f++) {
            cumulative = cumulative.concat(frameDetections[f]);
            detectedAtFrame[f] = [...cumulative];
        }

        // Progressive heatmap: builds up as detections accumulate
        for (let f = 0; f < totalFrames; f++) {
            const tempHeatmap = new Float32Array(GRID * GRID);
            const sigma = 1.5;
            for (const det of detectedAtFrame[f]) {
                if (det.confidence < 0.5) continue;
                for (let dr = -3; dr <= 3; dr++) {
                    for (let dc = -3; dc <= 3; dc++) {
                        const nr = det.row + dr;
                        const nc = det.col + dc;
                        if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) continue;
                        const dist2 = dr * dr + dc * dc;
                        tempHeatmap[nr * GRID + nc] += det.confidence * Math.exp(-dist2 / (2 * sigma * sigma));
                    }
                }
            }
            // Normalize against global max
            let maxV = 0;
            for (let i = 0; i < tempHeatmap.length; i++) if (tempHeatmap[i] > maxV) maxV = tempHeatmap[i];
            if (maxV > 0) for (let i = 0; i < tempHeatmap.length; i++) tempHeatmap[i] /= maxV;
            heatmapAtFrame[f] = tempHeatmap;
        }

        $('frame-counter').textContent = `Frame 0 / ${totalFrames}`;
    }

    // ── Animation ─────────────────────────────────────────────────────────────
    function togglePlay() {
        if (isPlaying) {
            pause();
        } else {
            play();
        }
    }

    function play() {
        if (currentFrame >= totalFrames - 1) currentFrame = 0;
        isPlaying = true;
        updatePlayIcon();
        animTimer = setInterval(stepForward, 1000 / fps);
    }

    function pause() {
        isPlaying = false;
        updatePlayIcon();
        clearInterval(animTimer);
        animTimer = null;
    }

    function resetSimulation() {
        pause();
        currentFrame = 0;
        visitedCells = new Set();
        sprayFlashes = [];
        draw();
        updateMetricsForFrame(0);
        $('progress-fill').style.width = '0%';
        $('frame-counter').textContent = `Frame 0 / ${totalFrames}`;
    }

    function stepForward() {
        if (currentFrame >= totalFrames - 1) {
            pause();
            return;
        }
        currentFrame++;
        const p = astarPath[currentFrame];
        visitedCells.add(`${p.row},${p.col}`);

        // Check spray flash
        const idx = p.row * GRID + p.col;
        if (sprayDecisions[idx] === 'spray') {
            sprayFlashes.push({
                cx: (p.col + 0.5) * cellW,
                cy: (p.row + 0.5) * cellH,
                frame: 0,
                maxFrames: 12,
            });
        }

        draw();
        updateMetricsForFrame(currentFrame);

        // Progress bar
        const pct = (currentFrame / (totalFrames - 1)) * 100;
        $('progress-fill').style.width = pct + '%';
        $('frame-counter').textContent = `Frame ${currentFrame} / ${totalFrames}`;
    }

    function updatePlayIcon() {
        $('icon-play').style.display = isPlaying ? 'none' : 'block';
        $('icon-pause').style.display = isPlaying ? 'block' : 'none';
    }

    // ── Metrics Update ────────────────────────────────────────────────────────
    function updateMetricsForFrame(frame) {
        if (frame < 0 || frame >= totalFrames) return;

        // Frame 0: show zeros for all metrics except latency (default 38ms)
        if (frame === 0) {
            $('metric-weeds').textContent = '0';
            $('metric-sprayed').textContent = '0';
            $('metric-skipped').textContent = '0';
            $('metric-chemical').innerHTML = '0.0<small>%</small>';
            $('metric-coverage').innerHTML = '0.0<small>%</small>';
            $('metric-pathlen').innerHTML = '0.0<small>m</small>';
            $('metric-latency').innerHTML = '38<small>ms</small>';
            $('metric-replans').textContent = '0';
            return;
        }

        // ─── REAL DATA MODE: read metrics directly from frame ────────────
        if (realScenarioData && realScenarioData.frames && frame < realScenarioData.frames.length) {
            const m = realScenarioData.frames[frame].metrics;
            $('metric-weeds').textContent = m.weeds_detected;
            $('metric-sprayed').textContent = m.cells_sprayed;
            $('metric-skipped').textContent = m.cells_skipped;
            $('metric-chemical').innerHTML = m.chemical_saved_pct.toFixed(1) + '<small>%</small>';
            $('metric-coverage').innerHTML = Math.min(m.coverage_pct, 100).toFixed(1) + '<small>%</small>';
            $('metric-pathlen').innerHTML = m.path_length_m.toFixed(1) + '<small>m</small>';
            $('metric-latency').innerHTML = m.avg_latency_ms.toFixed(0) + '<small>ms</small>';
            $('metric-replans').textContent = Math.min(m.replanning_events, 8);
            return;
        }

        // ─── PROCEDURAL MODE: compute metrics from simulation state ──────
        const dets = detectedAtFrame[frame] || [];
        const weedCount = dets.filter(d => d.confidence > 0.5).length;

        // Count unique visited cells up to this frame
        const totalTraversable = boustrophedonPath.length;
        const uniqueVisited = new Set();
        for (let f = 0; f <= frame && f < astarPath.length; f++) {
            const p = astarPath[f];
            uniqueVisited.add(`${p.row},${p.col}`);
        }

        // Spray/skip counts for visited cells only
        let sprayed = 0, skipped = 0;
        for (const key of uniqueVisited) {
            const [rs, cs] = key.split(',').map(Number);
            const idx = rs * GRID + cs;
            if (sprayDecisions[idx] === 'spray') sprayed++;
            else if (sprayDecisions[idx] === 'skip') skipped++;
        }

        // Chemical saved = (cells skipped / total traversable cells) × 100
        const totalCells = totalTraversable;
        const chemSaved = totalCells > 0 ? (skipped / totalCells * 100) : 0;

        // Coverage: starts at 0%, reaches 100% only at final frame
        const coverage = totalTraversable > 0 ? (uniqueVisited.size / totalTraversable * 100) : 0;

        // Path length = unique cells visited × 0.5 metres (cell size), capped at 200m
        const pathLen = Math.min(uniqueVisited.size * CELL_SIZE_M, 200.0);

        // Simulated latency (38ms ± noise)
        const latency = 38 + (Math.sin(frame * 0.3) * 8);

        // Replanning events — capped at 8
        const replanInterval = Math.max(1, Math.floor(totalFrames / 6));
        const replans = Math.min(Math.floor(frame / replanInterval), 8);

        $('metric-weeds').textContent = weedCount;
        $('metric-sprayed').textContent = sprayed;
        $('metric-skipped').textContent = skipped;
        $('metric-chemical').innerHTML = chemSaved.toFixed(1) + '<small>%</small>';
        $('metric-coverage').innerHTML = Math.min(coverage, 100).toFixed(1) + '<small>%</small>';
        $('metric-pathlen').innerHTML = pathLen.toFixed(1) + '<small>m</small>';
        $('metric-latency').innerHTML = latency.toFixed(0) + '<small>ms</small>';
        $('metric-replans').textContent = replans;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  DRAWING
    // ═══════════════════════════════════════════════════════════════════════════

    function draw() {
        if (!bgImage) return;
        ctx.clearRect(0, 0, imgW, imgH);

        // Background image
        ctx.drawImage(bgImage, 0, 0, imgW, imgH);

        const frame = currentFrame;
        const currentHeatmap = (frame < heatmapAtFrame.length) ? heatmapAtFrame[frame] : heatmap;
        const currentDetections = (frame < detectedAtFrame.length) ? detectedAtFrame[frame] : detections;

        // Layer 2: Heatmap overlay
        if (layers.heatmap) drawHeatmap(currentHeatmap);

        // Layer 4: Spray decisions
        if (layers.spray) drawSprayDecisions(currentHeatmap, currentDetections);

        // Layer 3: Navigation paths
        if (layers.path) drawPaths();

        // Layer 1: Detection bounding boxes
        if (layers.detection) drawDetections(currentDetections);

        // Layer 5: Robot
        if (layers.robot) drawRobot();

        // Spray flashes
        drawSprayFlashes();

        // Grid overlay (subtle)
        drawGrid();
    }

    // ── Layer 1: Detection Boxes ──────────────────────────────────────────────
    function drawDetections(dets) {
        for (const det of dets) {
            if (det.confidence < 0.5) continue;
            const color = CLASS_COLORS[det.classId % CLASS_COLORS.length];
            const isHighConf = det.confidence > 0.85;

            // Box size: clamp to min 30px, max 80px, and at most 1 grid cell
            let bw = Math.min(cellW, 80);
            let bh = Math.min(cellH, 80);
            bw = Math.max(bw, 30);
            bh = Math.max(bh, 30);

            // Center box on detection point within the cell
            const cx = (det.col + 0.5) * cellW;
            const cy = (det.row + 0.5) * cellH;
            const x = cx - bw / 2;
            const y = cy - bh / 2;

            // Box — 3px for high-confidence (>0.85), 2px for normal
            ctx.strokeStyle = color;
            ctx.lineWidth = isHighConf ? 3 : 2;
            ctx.setLineDash([]);

            // Subtle glow (brighter for high confidence)
            ctx.shadowColor = color;
            ctx.shadowBlur = isHighConf ? 8 : 4;
            ctx.strokeRect(x, y, bw, bh);
            ctx.shadowBlur = 0;

            // Confidence label — 10px font, top-left corner
            const label = det.confidence.toFixed(2);
            ctx.font = "600 10px 'Inter', sans-serif";
            const tw = ctx.measureText(label).width;
            const labelX = x;
            const labelY = y - 2;

            // Background pill — matches border color at 80% opacity
            ctx.fillStyle = hexToRgba(color, 0.8);
            const pillH = 13;
            const pillW = tw + 6;
            roundRect(ctx, labelX, labelY - pillH, pillW, pillH, 2);
            ctx.fill();

            // Text
            ctx.fillStyle = '#000';
            ctx.fillText(label, labelX + 3, labelY - 3);
        }
    }

    /**
     * Convert a hex color string (#RRGGBB) to rgba() with given alpha.
     */
    function hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    // ── Layer 2: Heatmap Overlay ──────────────────────────────────────────────
    function drawHeatmap(hm) {
        for (let r = 0; r < GRID; r++) {
            for (let c = 0; c < GRID; c++) {
                const val = hm[r * GRID + c];
                if (val < 0.01) continue;

                const color = heatmapColor(val);
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.4 * val;
                ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
                ctx.globalAlpha = 1.0;
            }
        }
    }

    function heatmapColor(val) {
        // Blue → Yellow → Red gradient
        let r, g, b;
        if (val < 0.5) {
            const t = val / 0.5;
            r = Math.floor(21 + t * (255 - 21));    // 21 → 255
            g = Math.floor(101 + t * (235 - 101));   // 101 → 235
            b = Math.floor(192 - t * (192 - 59));    // 192 → 59
        } else {
            const t = (val - 0.5) / 0.5;
            r = Math.floor(255 - t * (255 - 229));   // 255 → 229
            g = Math.floor(235 - t * (235 - 57));    // 235 → 57
            b = Math.floor(59 - t * 59);             // 59 → 0
        }
        return `rgb(${r},${g},${b})`;
    }

    // ── Layer 3: Paths ────────────────────────────────────────────────────────
    function drawPaths() {
        // Boustrophedon: clean serpentine S-pattern with perfectly parallel rows
        if (boustrophedonPath.length > 1) {
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = 'rgba(158, 158, 158, 0.45)';
            ctx.lineWidth = 1.2;
            ctx.lineCap = 'round';
            ctx.beginPath();

            // Draw row by row: horizontal lines with vertical connectors
            let prevRow = -1;
            let segStart = null;
            for (let i = 0; i < boustrophedonPath.length; i++) {
                const p = boustrophedonPath[i];
                const px = (p.col + 0.5) * cellW;
                const py = (p.row + 0.5) * cellH;

                if (p.row !== prevRow) {
                    // New row — if there was a previous segment, draw connector
                    if (segStart !== null) {
                        ctx.lineTo(px, py);
                    } else {
                        ctx.moveTo(px, py);
                    }
                    segStart = { x: px, y: py };
                    prevRow = p.row;
                } else {
                    ctx.lineTo(px, py);
                }
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // A* path: smooth bezier curves, animated up to current frame
        if (astarPath.length > 1 && currentFrame > 0) {
            const drawUpTo = Math.min(currentFrame, astarPath.length - 1);

            // Helper: get pixel position for path index
            const px = (i) => (astarPath[i].col + 0.5) * cellW;
            const py = (i) => (astarPath[i].row + 0.5) * cellH;

            // Trail glow (wider, subtle)
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(66, 165, 245, 0.12)';
            ctx.lineWidth = 6;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.moveTo(px(0), py(0));
            for (let i = 1; i <= drawUpTo; i++) {
                if (i < drawUpTo) {
                    // Quadratic bezier: control point is current, end is midpoint to next
                    const mx = (px(i) + px(i + 1 <= drawUpTo ? i + 1 : i)) / 2;
                    const my = (py(i) + py(i + 1 <= drawUpTo ? i + 1 : i)) / 2;
                    ctx.quadraticCurveTo(px(i), py(i), mx, my);
                } else {
                    ctx.lineTo(px(i), py(i));
                }
            }
            ctx.stroke();

            // Core line — bright blue, solid
            ctx.beginPath();
            ctx.strokeStyle = '#42a5f5';
            ctx.lineWidth = 2.5;
            ctx.setLineDash([]);
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.moveTo(px(0), py(0));
            for (let i = 1; i <= drawUpTo; i++) {
                if (i < drawUpTo) {
                    const mx = (px(i) + px(i + 1 <= drawUpTo ? i + 1 : i)) / 2;
                    const my = (py(i) + py(i + 1 <= drawUpTo ? i + 1 : i)) / 2;
                    ctx.quadraticCurveTo(px(i), py(i), mx, my);
                } else {
                    ctx.lineTo(px(i), py(i));
                }
            }
            ctx.stroke();
        }
    }

    // ── Layer 4: Spray Decision Markers ───────────────────────────────────────
    function drawSprayDecisions(hm, dets) {
        for (let r = 1; r < GRID - 1; r++) {
            for (let c = 1; c < GRID - 1; c++) {
                const idx = r * GRID + c;
                if (obstacles[idx]) continue;

                // Only show spray decisions for visited cells
                const visited = visitedCells.has(`${r},${c}`) || currentFrame === 0;
                if (!visited && currentFrame > 0) continue;

                const cx = (c + 0.5) * cellW;
                const cy = (r + 0.5) * cellH;

                const density = hm[idx];
                const hasConfident = dets.some(d => d.row === r && d.col === c && d.confidence > tauC);

                if (density > tauD && hasConfident) {
                    // Spray: dark blue circle
                    ctx.beginPath();
                    ctx.arc(cx, cy, Math.min(cellW, cellH) * 0.2, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(13, 71, 161, 0.7)';
                    ctx.fill();
                    ctx.strokeStyle = '#0d47a1';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                } else {
                    // Skip: small grey dot
                    ctx.beginPath();
                    ctx.arc(cx, cy, Math.min(cellW, cellH) * 0.08, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(97, 97, 97, 0.4)';
                    ctx.fill();
                }
            }
        }
    }

    // ── Layer 5: Robot ────────────────────────────────────────────────────────
    function drawRobot() {
        if (astarPath.length === 0) return;
        const idx = Math.min(currentFrame, astarPath.length - 1);
        const p = astarPath[idx];

        // Calculate angle to next cell
        let angle = 0;
        if (idx < astarPath.length - 1) {
            const next = astarPath[idx + 1];
            angle = Math.atan2((next.row - p.row) * cellH, (next.col - p.col) * cellW);
        } else if (idx > 0) {
            const prev = astarPath[idx - 1];
            angle = Math.atan2((p.row - prev.row) * cellH, (p.col - prev.col) * cellW);
        }

        const cx = (p.col + 0.5) * cellW;
        const cy = (p.row + 0.5) * cellH;
        const radius = Math.min(cellW, cellH) * 0.35;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angle);

        // Glow
        ctx.shadowColor = '#ce93d8';
        ctx.shadowBlur = 15;

        // Body circle
        ctx.beginPath();
        ctx.arc(0, 0, radius, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
        grad.addColorStop(0, 'rgba(206, 147, 216, 0.9)');
        grad.addColorStop(1, 'rgba(156, 39, 176, 0.7)');
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.strokeStyle = '#ce93d8';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.shadowBlur = 0;

        // Arrow
        ctx.beginPath();
        ctx.moveTo(radius * 0.7, 0);
        ctx.lineTo(-radius * 0.3, -radius * 0.4);
        ctx.lineTo(-radius * 0.3, radius * 0.4);
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.fill();

        ctx.restore();

        // Scan ring animation
        const pulseR = radius * (1.3 + 0.3 * Math.sin(currentFrame * 0.3));
        ctx.beginPath();
        ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(206, 147, 216, 0.25)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // ── Spray Flashes ─────────────────────────────────────────────────────────
    function drawSprayFlashes() {
        const remaining = [];
        for (const flash of sprayFlashes) {
            flash.frame++;
            if (flash.frame > flash.maxFrames) continue;

            const progress = flash.frame / flash.maxFrames;
            const r = Math.min(cellW, cellH) * (0.3 + progress * 1.0);
            const alpha = 0.6 * (1 - progress);

            ctx.beginPath();
            ctx.arc(flash.cx, flash.cy, r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(13, 71, 161, ${alpha})`;
            ctx.fill();

            // Inner circle
            const r2 = r * 0.4;
            ctx.beginPath();
            ctx.arc(flash.cx, flash.cy, r2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(30, 58, 138, ${alpha * 1.5})`;
            ctx.fill();

            remaining.push(flash);
        }
        sprayFlashes.length = 0;
        sprayFlashes.push(...remaining.filter(f => f.frame <= f.maxFrames));
    }

    // ── Grid Overlay ──────────────────────────────────────────────────────────
    function drawGrid() {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
        ctx.lineWidth = 0.5;
        ctx.setLineDash([]);
        for (let r = 0; r <= GRID; r++) {
            ctx.beginPath();
            ctx.moveTo(0, r * cellH);
            ctx.lineTo(imgW, r * cellH);
            ctx.stroke();
        }
        for (let c = 0; c <= GRID; c++) {
            ctx.beginPath();
            ctx.moveTo(c * cellW, 0);
            ctx.lineTo(c * cellW, imgH);
            ctx.stroke();
        }

        // Obstacle cells
        for (let r = 0; r < GRID; r++) {
            for (let c = 0; c < GRID; c++) {
                if (obstacles[r * GRID + c]) {
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                    ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
                    // Cross
                    ctx.strokeStyle = 'rgba(239, 83, 80, 0.5)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(c * cellW + 3, r * cellH + 3);
                    ctx.lineTo((c + 1) * cellW - 3, (r + 1) * cellH - 3);
                    ctx.moveTo((c + 1) * cellW - 3, r * cellH + 3);
                    ctx.lineTo(c * cellW + 3, (r + 1) * cellH - 3);
                    ctx.stroke();
                }
            }
        }
    }

    // ── Utility: Rounded Rectangle ────────────────────────────────────────────
    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    // ── Init ──────────────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', init);
})();

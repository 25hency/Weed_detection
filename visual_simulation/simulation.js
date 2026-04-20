(() => {
    'use strict';

    const GRID = 20;
    const CELL_SIZE_M = 0.5;
    const TAU_D = 0.3;
    const TAU_C = 0.5;
    const NUM_CLASSES = 15;
    const REPLANNING_HEATMAP_DELTA_THRESHOLD = 0.15;

    const CLASS_COLORS = [
        '#EF4444', '#F97316', '#EAB308', '#84CC16', '#06B6D4',
        '#38BDF8', '#A855F7', '#EC4899', '#F87171', '#14B8A6',
        '#F59E0B', '#8B5CF6', '#10B981', '#FB7185', '#6366F1',
    ];

    const SCENARIO_CFG = {
        low:         { weedMin: 20, weedMax: 30, clusters: 2, radius: 1.5, noise: 1.0 },
        medium:      { weedMin: 35, weedMax: 45, clusters: 4, radius: 2.0, noise: 1.0 },
        high:        { weedMin: 50, weedMax: 65, clusters: 6, radius: 2.5, noise: 1.0 },
        shadowed:    { weedMin: 30, weedMax: 40, clusters: 3, radius: 2.0, noise: 2.0 },
        overlapping: { weedMin: 40, weedMax: 55, clusters: 5, radius: 3.0, noise: 1.0 },
    };

    const SCENARIO_KEY_MAP = {
        'low': 'LOW_DENSITY', 'medium': 'MEDIUM_DENSITY', 'high': 'HIGH_DENSITY',
        'shadowed': 'SHADOWED', 'overlapping': 'OVERLAPPING',
    };

    function mulberry32(a) {
        return function () {
            let t = a += 0x6D2B79F5;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    let rng = mulberry32(42);
    let canvas, ctx;
    let bgImage = null;
    let imgW = 0, imgH = 0;
    let cellW = 0, cellH = 0;

    let detections = [];
    let heatmap = [];
    let astarPath = [];
    let astarDrawPath = [];
    let boustrophedonPath = [];
    let sprayDecisions = [];
    let obstacles = [];

    let currentFrame = 0;
    let totalFrames = 0;
    let isPlaying = false;
    let animTimer = null;
    let sprayFlashes = [];
    let visitedCells = new Set();
    let detectedAtFrame = [];
    let heatmapAtFrame = [];

    let realData = null;
    let realScenarioData = null;
    let useRealData = false;
    let previousHeatmap = [];

    let layers = { detection: true, heatmap: true, path: true, spray: true, robot: true };
    let tauD = TAU_D;
    let tauC = TAU_C;
    let fps = 15;

    const $ = (id) => document.getElementById(id);

    function generateGridBackground() {
        const container = $('canvas-container');
        const size = Math.min(container.clientWidth - 20, container.clientHeight - 20, 800);
        imgW = size;
        imgH = size;
        canvas.width = imgW;
        canvas.height = imgH;
        cellW = imgW / GRID;
        cellH = imgH / GRID;

        const offscreen = document.createElement('canvas');
        offscreen.width = imgW;
        offscreen.height = imgH;
        const octx = offscreen.getContext('2d');

        const grad = octx.createLinearGradient(0, 0, imgW, imgH);
        grad.addColorStop(0, '#3d2e1f');
        grad.addColorStop(0.5, '#4a3728');
        grad.addColorStop(1, '#3d2e1f');
        octx.fillStyle = grad;
        octx.fillRect(0, 0, imgW, imgH);

        for (let r = 1; r < GRID - 1; r++) {
            const y = r * cellH;
            octx.fillStyle = r % 2 === 0 ? '#2d5016' : '#3a6b20';
            octx.globalAlpha = 0.3;
            octx.fillRect(0, y, imgW, cellH);
        }
        octx.globalAlpha = 1.0;

        octx.strokeStyle = 'rgba(255,255,255,0.08)';
        octx.lineWidth = 0.5;
        for (let r = 0; r <= GRID; r++) {
            octx.beginPath(); octx.moveTo(0, r * cellH); octx.lineTo(imgW, r * cellH); octx.stroke();
        }
        for (let c = 0; c <= GRID; c++) {
            octx.beginPath(); octx.moveTo(c * cellW, 0); octx.lineTo(c * cellW, imgH); octx.stroke();
        }

        const dataUrl = offscreen.toDataURL();
        bgImage = new Image();
        bgImage.src = dataUrl;

        canvas.classList.add('active');
        $('placeholder-msg').style.display = 'none';
    }

    async function autoLoadVisualData() {
        try {
            const resp = await fetch('visual_data.json');
            if (!resp.ok) return false;
            const data = await resp.json();
            realData = data;
            useRealData = true;
            const status = $('json-status');
            const scenarioCount = Object.keys(data.scenarios || {}).length;
            if (status) {
                status.textContent = `✓ ${scenarioCount} scenarios loaded`;
                status.className = 'json-status loaded';
            }
            return true;
        } catch {
            return false;
        }
    }

    async function init() {
        canvas = $('sim-canvas');
        ctx = canvas.getContext('2d');

        $('upload-btn').addEventListener('click', () => $('file-input').click());
        $('file-input').addEventListener('change', handleFileUpload);

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

        $('btn-play').addEventListener('click', togglePlay);
        $('btn-reset').addEventListener('click', resetSimulation);
        $('btn-step').addEventListener('click', stepForward);

        $('speed-slider').addEventListener('input', (e) => {
            fps = parseInt(e.target.value);
            $('speed-value').textContent = fps + ' fps';
            if (isPlaying) { clearInterval(animTimer); animTimer = setInterval(stepForward, 1000 / fps); }
        });

        $('scenario-select').addEventListener('change', () => {
            if (bgImage) regenerateAndReset();
        });

        ['detection', 'heatmap', 'path', 'spray', 'robot'].forEach(id => {
            $('layer-' + id).addEventListener('change', (e) => {
                layers[id] = e.target.checked;
                draw();
            });
        });

        await autoLoadVisualData();
    }

    function handleJsonFile(file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
            try {
                realData = JSON.parse(ev.target.result);
                useRealData = true;
                const status = $('json-status');
                const scenarioCount = Object.keys(realData.scenarios || {}).length;
                if (status) {
                    status.textContent = `✓ ${scenarioCount} scenarios loaded`;
                    status.className = 'json-status loaded';
                }
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
                canvas.classList.add('active');
                $('placeholder-msg').style.display = 'none';
                regenerateAndReset();
            };
            img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
    }

    function regenerateAndReset() {
        realScenarioData = useRealData ? getActiveRealScenario() : null;
        rng = mulberry32(Date.now() & 0xFFFFFFFF);
        const scenario = $('scenario-select').value;
        const cfg = SCENARIO_CFG[scenario];

        heatmap = new Float32Array(GRID * GRID);
        sprayDecisions = new Array(GRID * GRID).fill(null);
        obstacles = new Uint8Array(GRID * GRID);
        detections = [];
        visitedCells = new Set();
        sprayFlashes = [];

        if (realScenarioData && realScenarioData.frames && realScenarioData.frames.length > 0) {
            loadRealScenarioData();
        } else {
            generateObstacles();
            generateDetections(cfg);
            computeHeatmap();
            recomputeSprayDecisions();
            generateBoustrophedonPath();
            generateAStarPath();
            precomputeFrames();
        }

        $('btn-play').disabled = false;
        $('btn-reset').disabled = false;
        $('btn-step').disabled = false;

        currentFrame = 0;
        isPlaying = false;
        updatePlayIcon();
        draw();
        updateMetricsForFrame(0);
    }

    function loadRealScenarioData() {
        const sd = realScenarioData;
        const frames = sd.frames;
        totalFrames = frames.length;

        if (sd.obstacles) {
            for (let r = 0; r < GRID; r++)
                for (let c = 0; c < GRID; c++)
                    obstacles[r * GRID + c] = sd.obstacles[r][c] ? 1 : 0;
        } else {
            generateObstacles();
        }

        if (sd.boustrophedon_path) {
            boustrophedonPath = sd.boustrophedon_path.map(p => ({ row: p.row, col: p.col }));
        } else {
            generateBoustrophedonPath();
        }

        const allDets = new Map();
        for (const frame of frames) {
            for (const d of frame.detections) {
                const key = `${d.cell_i},${d.cell_j},${d.class_id}`;
                if (!allDets.has(key) || d.confidence > allDets.get(key).confidence) {
                    allDets.set(key, d);
                }
            }
        }
        detections = Array.from(allDets.values()).map(d => ({
            row: d.cell_i, col: d.cell_j, classId: d.class_id, confidence: d.confidence,
        }));

        computeHeatmap();

        sprayDecisions = new Array(GRID * GRID).fill('skip');
        for (const frame of frames) {
            if (frame.spray_decision) {
                const idx = frame.cell_row * GRID + frame.cell_col;
                sprayDecisions[idx] = 'spray';
            }
        }

        generateAStarPath();
        precomputeFrames();
    }

    function generateObstacles() {
        for (let i = 0; i < GRID; i++) {
            obstacles[0 * GRID + i] = 1;
            obstacles[(GRID - 1) * GRID + i] = 1;
            obstacles[i * GRID + 0] = 1;
            obstacles[i * GRID + (GRID - 1)] = 1;
        }
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
        if (cfg.clusters <= 0) return;

        const clusterCenters = [];
        for (let i = 0; i < cfg.clusters; i++) {
            clusterCenters.push({
                r: 2 + Math.floor(rng() * (GRID - 4)),
                c: 2 + Math.floor(rng() * (GRID - 4)),
                classId: i % 6,
            });
        }

        for (let k = 0; k < count; k++) {
            const center = clusterCenters[Math.floor(rng() * clusterCenters.length)];
            let r, c, attempts = 0;
            do {
                const dr = gaussianRng() * cfg.radius;
                const dc = gaussianRng() * cfg.radius;
                r = Math.round(center.r + dr);
                c = Math.round(center.c + dc);
                attempts++;
            } while ((r < 1 || r >= GRID - 1 || c < 1 || c >= GRID - 1 || obstacles[r * GRID + c]) && attempts < 50);
            if (attempts >= 50) continue;

            let nearestCluster = null, nearestDist = Infinity;
            for (const cc of clusterCenters) {
                const dist = Math.sqrt((r - cc.r) ** 2 + (c - cc.c) ** 2);
                if (dist <= 3 && dist < nearestDist) { nearestDist = dist; nearestCluster = cc; }
            }
            const classId = nearestCluster ? nearestCluster.classId : (6 + k % 9);

            let conf = 0.5 + rng() * 0.5;
            if (cfg.noise > 1) conf *= (0.7 + rng() * 0.3);
            conf = Math.min(1.0, Math.max(0.35, conf));

            detections.push({ row: r, col: c, classId, confidence: conf });
        }
    }

    function gaussianRng() {
        let u = 0, v = 0;
        while (u === 0) u = rng();
        while (v === 0) v = rng();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function computeHeatmap() {
        heatmap = new Float32Array(GRID * GRID);
        const sigma = 1.0;
        for (const det of detections) {
            if (det.confidence < 0.5) continue;
            for (let dr = -2; dr <= 2; dr++) {
                for (let dc = -2; dc <= 2; dc++) {
                    const nr = det.row + dr, nc = det.col + dc;
                    if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) continue;
                    heatmap[nr * GRID + nc] += det.confidence * Math.exp(-(dr * dr + dc * dc) / (2 * sigma * sigma));
                }
            }
        }
        let maxVal = 0;
        for (let i = 0; i < heatmap.length; i++) if (heatmap[i] > maxVal) maxVal = heatmap[i];
        if (maxVal > 0) for (let i = 0; i < heatmap.length; i++) heatmap[i] /= maxVal;
    }

    function generateBoustrophedonPath() {
        boustrophedonPath = [];
        let leftToRight = true;
        for (let r = 1; r < GRID - 1; r++) {
            if (leftToRight) {
                for (let c = 1; c < GRID - 1; c++) { if (!obstacles[r * GRID + c]) boustrophedonPath.push({ row: r, col: c }); }
            } else {
                for (let c = GRID - 2; c >= 1; c--) { if (!obstacles[r * GRID + c]) boustrophedonPath.push({ row: r, col: c }); }
            }
            leftToRight = !leftToRight;
        }
    }

    function prunePath(path, epsilon = 0.5) {
        if (path.length <= 2) return path;
        function distPtLine(p, a, b) {
            const num = Math.abs((b.col - a.col) * (a.row - p.row) - (a.col - p.col) * (b.row - a.row));
            const den = Math.sqrt((b.col - a.col) ** 2 + (b.row - a.row) ** 2);
            return den > 0 ? num / den : 0;
        }
        function rdp(pts, eps) {
            let maxD = 0, maxI = 0;
            for (let i = 1; i < pts.length - 1; i++) {
                const d = distPtLine(pts[i], pts[0], pts[pts.length - 1]);
                if (d > maxD) { maxD = d; maxI = i; }
            }
            if (maxD > eps) {
                const left = rdp(pts.slice(0, maxI + 1), eps);
                const right = rdp(pts.slice(maxI), eps);
                return left.slice(0, -1).concat(right);
            }
            return [pts[0], pts[pts.length - 1]];
        }
        return rdp(path, epsilon);
    }

    function lineOfSightSimplify(path) {
        if (path.length <= 2) return path;
        const simplified = [path[0]];
        for (let i = 1; i < path.length; i++) {
            const last = simplified[simplified.length - 1];
            const current = path[i];
            let canSkip = true;
            const dx = Math.abs(current.col - last.col);
            const dy = Math.abs(current.row - last.row);
            const sx = last.col < current.col ? 1 : -1;
            const sy = last.row < current.row ? 1 : -1;
            let err = (dx > dy ? dx : -dy) / 2;
            let x = last.col, y = last.row;
            while (x !== current.col || y !== current.row) {
                if (obstacles[y * GRID + x]) { canSkip = false; break; }
                const e = err;
                if (e > -dx) { err -= dy; x += sx; }
                if (e < dy) { err += dx; y += sy; }
            }
            if (!canSkip) simplified.push(path[i - 1]);
        }
        simplified.push(path[path.length - 1]);
        return simplified;
    }

    function generateAStarPath() {
        const sprayCells = [];
        for (let r = 1; r < GRID - 1; r++) {
            for (let c = 1; c < GRID - 1; c++) {
                const idx = r * GRID + c;
                if (sprayDecisions[idx] === 'spray') {
                    sprayCells.push({ row: r, col: c, val: heatmap[idx] });
                }
            }
        }
        sprayCells.sort((a, b) => b.val - a.val);

        astarPath = [];
        const globalVisited = new Set();
        let current = { row: 1, col: 1 };
        astarPath.push(current);
        globalVisited.add(`${current.row},${current.col}`);

        const remaining = [...sprayCells];
        while (remaining.length > 0) {
            let bestIdx = 0, bestDist = Infinity;
            for (let i = 0; i < remaining.length; i++) {
                const d = Math.abs(remaining[i].row - current.row) + Math.abs(remaining[i].col - current.col);
                const eff = d - remaining[i].val * 2;
                if (eff < bestDist) { bestDist = eff; bestIdx = i; }
            }
            const target = remaining.splice(bestIdx, 1)[0];
            if (globalVisited.has(`${target.row},${target.col}`)) continue;
            const seg = astarSegment(current, target);
            for (let i = 1; i < seg.length; i++) {
                const sk = `${seg[i].row},${seg[i].col}`;
                if (!globalVisited.has(sk)) { astarPath.push(seg[i]); globalVisited.add(sk); }
            }
            current = target;
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
                const path = [{ row: goal.row, col: goal.col }];
                let k = goalK;
                while (cameFrom.has(k)) { k = cameFrom.get(k); path.push({ row: Math.floor(k / GRID), col: k % GRID }); }
                path.reverse();
                return path;
            }
            if (closed.has(ck)) continue;
            closed.add(ck);
            for (const [dr, dc] of dirs) {
                const nr = cur.row + dr, nc = cur.col + dc;
                if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) continue;
                const nk = key(nr, nc);
                if (closed.has(nk) || obstacles[nk]) continue;
                const dist = Math.sqrt(dr * dr + dc * dc) * CELL_SIZE_M;
                const hVal = heatmap[nk] || 0;
                const stepCost = Math.max(0.01, dist - 0.6 * hVal * CELL_SIZE_M);
                const tentG = (gScore.get(ck) || 0) + stepCost;
                if (tentG < (gScore.get(nk) || Infinity)) {
                    gScore.set(nk, tentG);
                    cameFrom.set(nk, ck);
                    open.push({ row: nr, col: nc, f: tentG + heuristic({ row: nr, col: nc }, goal) });
                }
            }
        }
        return [start];
    }

    function heuristic(a, b) {
        const dr = Math.abs(a.row - b.row), dc = Math.abs(a.col - b.col);
        return CELL_SIZE_M * (Math.max(dr, dc) + (Math.SQRT2 - 1) * Math.min(dr, dc));
    }

    class MinHeap {
        constructor() { this.data = []; }
        size() { return this.data.length; }
        push(val) { this.data.push(val); this._up(this.data.length - 1); }
        pop() {
            const top = this.data[0]; const last = this.data.pop();
            if (this.data.length > 0) { this.data[0] = last; this._down(0); }
            return top;
        }
        _up(i) {
            while (i > 0) {
                const p = (i - 1) >> 1;
                if (this.data[p].f <= this.data[i].f) break;
                [this.data[p], this.data[i]] = [this.data[i], this.data[p]]; i = p;
            }
        }
        _down(i) {
            const n = this.data.length;
            while (true) {
                let s = i; const l = 2 * i + 1, r = 2 * i + 2;
                if (l < n && this.data[l].f < this.data[s].f) s = l;
                if (r < n && this.data[r].f < this.data[s].f) s = r;
                if (s === i) break;
                [this.data[s], this.data[i]] = [this.data[i], this.data[s]]; i = s;
            }
        }
    }

    function recomputeSprayDecisions() {
        sprayDecisions = new Array(GRID * GRID).fill(null);
        for (let r = 1; r < GRID - 1; r++) {
            for (let c = 1; c < GRID - 1; c++) {
                const idx = r * GRID + c;
                if (obstacles[idx]) continue;
                const density = heatmap[idx];
                const hasConfident = detections.some(d => d.row === r && d.col === c && d.confidence > tauC);
                sprayDecisions[idx] = (density > tauD && hasConfident) ? 'spray' : 'skip';
            }
        }
    }

    function precomputeFrames() {
        totalFrames = astarPath.length;
        detectedAtFrame = new Array(totalFrames);
        heatmapAtFrame = new Array(totalFrames);

        const cellToFrameMap = new Map();
        for (let f = 0; f < astarPath.length; f++) {
            const p = astarPath[f];
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    const nk = `${p.row + dr},${p.col + dc}`;
                    if (!cellToFrameMap.has(nk)) cellToFrameMap.set(nk, f);
                }
            }
        }

        const frameDetections = new Array(totalFrames).fill(null).map(() => []);
        for (const det of detections) {
            const frame = cellToFrameMap.get(`${det.row},${det.col}`) || 0;
            frameDetections[Math.min(frame, totalFrames - 1)].push(det);
        }

        let cumulative = [];
        for (let f = 0; f < totalFrames; f++) {
            cumulative = cumulative.concat(frameDetections[f]);
            detectedAtFrame[f] = [...cumulative];
        }

        const sigma = 1.0;
        for (let f = 0; f < totalFrames; f++) {
            const tmp = new Float32Array(GRID * GRID);
            for (const det of detectedAtFrame[f]) {
                if (det.confidence < 0.5) continue;
                for (let dr = -2; dr <= 2; dr++) {
                    for (let dc = -2; dc <= 2; dc++) {
                        const nr = det.row + dr, nc = det.col + dc;
                        if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) continue;
                        tmp[nr * GRID + nc] += det.confidence * Math.exp(-(dr * dr + dc * dc) / (2 * sigma * sigma));
                    }
                }
            }
            let maxV = 0;
            for (let i = 0; i < tmp.length; i++) if (tmp[i] > maxV) maxV = tmp[i];
            if (maxV > 0) for (let i = 0; i < tmp.length; i++) tmp[i] /= maxV;
            heatmapAtFrame[f] = tmp;
        }

        $('frame-counter').textContent = `Frame 0 / ${totalFrames}`;
    }

    function togglePlay() { isPlaying ? pause() : play(); }

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
        if (currentFrame >= totalFrames - 1) { pause(); return; }
        currentFrame++;
        const p = astarPath[currentFrame];
        visitedCells.add(`${p.row},${p.col}`);
        const idx = p.row * GRID + p.col;
        if (sprayDecisions[idx] === 'spray') {
            sprayFlashes.push({ cx: (p.col + 0.5) * cellW, cy: (p.row + 0.5) * cellH, frame: 0, maxFrames: 12 });
        }
        draw();
        updateMetricsForFrame(currentFrame);
        const pct = (currentFrame / (totalFrames - 1)) * 100;
        $('progress-fill').style.width = pct + '%';
        $('frame-counter').textContent = `Frame ${currentFrame} / ${totalFrames}`;
    }

    function updatePlayIcon() {
        $('icon-play').style.display = isPlaying ? 'none' : 'block';
        $('icon-pause').style.display = isPlaying ? 'block' : 'none';
    }

    function updateMetricsForFrame(frame) {
        if (frame < 0 || frame >= totalFrames) return;
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

        const dets = detectedAtFrame[frame] || [];
        const weedCount = dets.filter(d => d.confidence > 0.5).length;
        const totalTraversable = boustrophedonPath.length;
        const uniqueVisited = new Set();
        for (let f = 0; f <= frame && f < astarPath.length; f++) {
            const p = astarPath[f];
            uniqueVisited.add(`${p.row},${p.col}`);
        }
        let sprayed = 0, skipped = 0;
        for (const key of uniqueVisited) {
            const [rs, cs] = key.split(',').map(Number);
            const idx = rs * GRID + cs;
            if (sprayDecisions[idx] === 'spray') sprayed++;
            else if (sprayDecisions[idx] === 'skip') skipped++;
        }
        const chemSaved = totalTraversable > 0 ? (skipped / totalTraversable * 100) : 0;
        const coverage = totalTraversable > 0 ? (uniqueVisited.size / totalTraversable * 100) : 0;
        const pathLen = Math.min(uniqueVisited.size * CELL_SIZE_M, 200.0);
        const latency = 38 + (Math.sin(frame * 0.3) * 8);
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

    function draw() {
        if (!bgImage) return;
        ctx.clearRect(0, 0, imgW, imgH);
        ctx.drawImage(bgImage, 0, 0, imgW, imgH);
        const frame = currentFrame;
        const currentHeatmap = (frame < heatmapAtFrame.length) ? heatmapAtFrame[frame] : heatmap;
        const currentDetections = (frame < detectedAtFrame.length) ? detectedAtFrame[frame] : detections;
        if (layers.heatmap) drawHeatmap(currentHeatmap);
        if (layers.spray) drawSprayDecisions();
        if (layers.path) drawPaths();
        if (layers.detection) drawDetections(currentDetections);
        if (layers.robot) drawRobot();
        drawSprayFlashes();
        drawGrid();
    }

    function drawDetections(dets) {
        for (const det of dets) {
            if (det.confidence < 0.5) continue;
            const color = CLASS_COLORS[det.classId % CLASS_COLORS.length];
            const isHigh = det.confidence > 0.85;
            let bw = Math.min(Math.max(cellW, 30), 80);
            let bh = Math.min(Math.max(cellH, 30), 80);
            const cx = (det.col + 0.5) * cellW, cy = (det.row + 0.5) * cellH;
            const x = cx - bw / 2, y = cy - bh / 2;
            ctx.strokeStyle = color;
            ctx.lineWidth = isHigh ? 3 : 2;
            ctx.setLineDash([]);
            ctx.shadowColor = color;
            ctx.shadowBlur = isHigh ? 8 : 4;
            ctx.strokeRect(x, y, bw, bh);
            ctx.shadowBlur = 0;
            const label = det.confidence.toFixed(2);
            ctx.font = "600 10px 'Inter', sans-serif";
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = hexToRgba(color, 0.8);
            roundRect(ctx, x, y - 15, tw + 6, 13, 2);
            ctx.fill();
            ctx.fillStyle = '#000';
            ctx.fillText(label, x + 3, y - 5);
        }
    }

    function hexToRgba(hex, alpha) {
        return `rgba(${parseInt(hex.slice(1,3),16)},${parseInt(hex.slice(3,5),16)},${parseInt(hex.slice(5,7),16)},${alpha})`;
    }

    function drawHeatmap(hm) {
        for (let r = 0; r < GRID; r++) {
            for (let c = 0; c < GRID; c++) {
                const val = hm[r * GRID + c];
                if (val < 0.01) continue;
                ctx.fillStyle = heatmapColor(val);
                ctx.globalAlpha = 0.4 * val;
                ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
                ctx.globalAlpha = 1.0;
            }
        }
    }

    function heatmapColor(val) {
        let r, g, b;
        if (val < 0.5) {
            const t = val / 0.5;
            r = Math.floor(21 + t * 234); g = Math.floor(101 + t * 134); b = Math.floor(192 - t * 133);
        } else {
            const t = (val - 0.5) / 0.5;
            r = Math.floor(255 - t * 26); g = Math.floor(235 - t * 178); b = Math.floor(59 - t * 59);
        }
        return `rgb(${r},${g},${b})`;
    }

    function drawPaths() {
        if (boustrophedonPath.length > 1) {
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = 'rgba(192,192,192,0.45)';
            ctx.lineWidth = 1.2;
            ctx.lineCap = 'round';
            ctx.beginPath();
            let prevRow = -1;
            for (let i = 0; i < boustrophedonPath.length; i++) {
                const p = boustrophedonPath[i];
                const px = (p.col + 0.5) * cellW, py = (p.row + 0.5) * cellH;
                if (p.row !== prevRow) { if (prevRow >= 0) ctx.lineTo(px, py); else ctx.moveTo(px, py); prevRow = p.row; }
                else ctx.lineTo(px, py);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }

        if (astarPath.length > 1 && currentFrame > 0) {
            const drawUpTo = Math.min(currentFrame, astarPath.length - 1);
            const px = (i) => (astarPath[i].col + 0.5) * cellW;
            const py = (i) => (astarPath[i].row + 0.5) * cellH;

            ctx.beginPath();
            ctx.strokeStyle = 'rgba(0,212,255,0.12)';
            ctx.lineWidth = 6; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
            ctx.moveTo(px(0), py(0));
            for (let i = 1; i <= drawUpTo; i++) ctx.lineTo(px(i), py(i));
            ctx.stroke();

            ctx.beginPath();
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2.5; ctx.setLineDash([]); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
            ctx.moveTo(px(0), py(0));
            for (let i = 1; i <= drawUpTo; i++) ctx.lineTo(px(i), py(i));
            ctx.stroke();
        }
    }

    function drawSprayDecisions() {
        if (currentFrame === 0) return;
        for (let r = 1; r < GRID - 1; r++) {
            for (let c = 1; c < GRID - 1; c++) {
                const idx = r * GRID + c;
                if (obstacles[idx]) continue;
                if (!visitedCells.has(`${r},${c}`)) continue;
                const cx = (c + 0.5) * cellW, cy = (r + 0.5) * cellH;
                if (sprayDecisions[idx] === 'spray') {
                    ctx.beginPath();
                    ctx.arc(cx, cy, Math.min(cellW, cellH) * 0.22, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(13,71,161,0.8)'; ctx.fill();
                    ctx.strokeStyle = '#1565c0'; ctx.lineWidth = 2; ctx.stroke();
                } else if (sprayDecisions[idx] === 'skip') {
                    ctx.beginPath();
                    ctx.arc(cx, cy, Math.min(cellW, cellH) * 0.06, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(97,97,97,0.3)'; ctx.fill();
                }
            }
        }
    }

    function drawRobot() {
        if (astarPath.length === 0) return;
        const idx = Math.min(currentFrame, astarPath.length - 1);
        const p = astarPath[idx];
        let angle = 0;
        if (idx < astarPath.length - 1) {
            const next = astarPath[idx + 1];
            angle = Math.atan2((next.row - p.row) * cellH, (next.col - p.col) * cellW);
        } else if (idx > 0) {
            const prev = astarPath[idx - 1];
            angle = Math.atan2((p.row - prev.row) * cellH, (p.col - prev.col) * cellW);
        }
        const cx = (p.col + 0.5) * cellW, cy = (p.row + 0.5) * cellH;
        const radius = Math.min(cellW, cellH) * 0.35;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angle);
        ctx.shadowColor = '#ce93d8'; ctx.shadowBlur = 15;
        ctx.beginPath(); ctx.arc(0, 0, radius, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
        grad.addColorStop(0, 'rgba(206,147,216,0.9)'); grad.addColorStop(1, 'rgba(156,39,176,0.7)');
        ctx.fillStyle = grad; ctx.fill();
        ctx.strokeStyle = '#ce93d8'; ctx.lineWidth = 2; ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.moveTo(radius * 0.7, 0); ctx.lineTo(-radius * 0.3, -radius * 0.4); ctx.lineTo(-radius * 0.3, radius * 0.4);
        ctx.closePath(); ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fill();
        ctx.restore();
        const pulseR = radius * (1.3 + 0.3 * Math.sin(currentFrame * 0.3));
        ctx.beginPath(); ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(206,147,216,0.25)'; ctx.lineWidth = 1; ctx.stroke();
    }

    function drawSprayFlashes() {
        const remaining = [];
        for (const flash of sprayFlashes) {
            flash.frame++;
            if (flash.frame > flash.maxFrames) continue;
            const progress = flash.frame / flash.maxFrames;
            const r = Math.min(cellW, cellH) * (0.3 + progress * 1.0);
            const alpha = 0.6 * (1 - progress);
            ctx.beginPath(); ctx.arc(flash.cx, flash.cy, r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(13,71,161,${alpha})`; ctx.fill();
            ctx.beginPath(); ctx.arc(flash.cx, flash.cy, r * 0.4, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(30,58,138,${alpha * 1.5})`; ctx.fill();
            remaining.push(flash);
        }
        sprayFlashes.length = 0;
        sprayFlashes.push(...remaining.filter(f => f.frame <= f.maxFrames));
    }

    function drawGrid() {
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 0.5; ctx.setLineDash([]);
        for (let r = 0; r <= GRID; r++) { ctx.beginPath(); ctx.moveTo(0, r * cellH); ctx.lineTo(imgW, r * cellH); ctx.stroke(); }
        for (let c = 0; c <= GRID; c++) { ctx.beginPath(); ctx.moveTo(c * cellW, 0); ctx.lineTo(c * cellW, imgH); ctx.stroke(); }
        for (let r = 0; r < GRID; r++) {
            for (let c = 0; c < GRID; c++) {
                if (obstacles[r * GRID + c]) {
                    ctx.fillStyle = 'rgba(0,0,0,0.3)';
                    ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
                    ctx.strokeStyle = 'rgba(239,83,80,0.5)'; ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(c * cellW + 3, r * cellH + 3); ctx.lineTo((c + 1) * cellW - 3, (r + 1) * cellH - 3);
                    ctx.moveTo((c + 1) * cellW - 3, r * cellH + 3); ctx.lineTo(c * cellW + 3, (r + 1) * cellH - 3);
                    ctx.stroke();
                }
            }
        }
    }

    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    document.addEventListener('DOMContentLoaded', init);
})();

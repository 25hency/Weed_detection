"""
Module 2 — YOLOv8 Weed Detection Inference Node  (AFRS Paper §3.3)

Confidence score (paper eq. 5):
    c_k = P(object) × P(class | object) × IoU(predicted, ground_truth)

Each component is drawn from scenario-appropriate Beta distributions:
    p_object              ~ Beta(8,2)  weed present  |  Beta(1,8) noise
    p_class_given_object  ~ Beta(7,2)  correct class |  Beta(2,5) wrong
    iou_score             ~ Beta(6,2)  well-localised|  Beta(2,4) occluded

SHADOWED scenario lowers p_object → Beta(5,3).
OVERLAPPING scenario lowers iou  → Beta(3,4).

NMS: IoU threshold = 0.45 (paper §3.3).

Class IDs:
    0–14 = weed species   (passed downstream)
    15   = crop           (filtered; must NOT contribute to heatmap)
    16   = background     (filtered)

Only detections with class_id in 0..14 AND confidence > τ_c (=0.5) are
published. Per-scenario detection statistics are logged.
"""

import time
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.node_base import NodeBase
from simulation.farm_world import (
    CELL_SIZE, SHADOWED, OVERLAPPING
)

logger = logging.getLogger(__name__)

TOPIC_CAMERA     = '/camera/image'
TOPIC_ODOM       = '/odom/pose'
TOPIC_DETECTIONS = '/detection/weeds'

# ── Constants ─────────────────────────────────────────────────────────────────
NMS_IOU_THRESHOLD    = 0.45   # paper §3.3
CONFIDENCE_THRESHOLD = 0.50   # τ_c (default); can be overridden per run
CLASS_ID_CROP        = 15
CLASS_ID_BACKGROUND  = 16
WEED_CLASS_RANGE     = range(0, 15)   # 0–14 inclusive


class DetectorNode(NodeBase):
    """
    YOLOv8 Weed Detection Inference Node.

    In simulation mode, generates structured synthetic detections from
    ground-truth camera data using Beta-distributed confidence components.
    Applies NMS and downstream filters before publishing.
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        use_simulation: bool = True,
        rate_hz: float = 30.0,
        scenario: str = "MEDIUM_DENSITY",
    ):
        super().__init__('detector_node', rate_hz=rate_hz)

        self.model_path    = model_path or str(Path(r"D:\Weed Detection\models\best.pt"))
        self.conf_threshold = confidence_threshold
        self.use_simulation = use_simulation
        self.scenario       = scenario

        self.model              = None
        self.latest_camera_msg  = None
        self.latest_odom_msg    = None
        self.detection_count    = 0
        self.inference_times: List[float] = []

        # Per-scenario stats
        self.total_detections_raw   = 0
        self.filtered_detections    = 0
        self.confidence_log: List[float] = []
        self.split_detection_count  = 0   # >1 detection for a single true cell

    # ── Node lifecycle ────────────────────────────────────────────────────────

    def on_start(self):
        if os.path.exists(self.model_path) and not self.use_simulation:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.model.fuse()
                logger.info(f"YOLOv8 loaded from {self.model_path}")
            except Exception as exc:
                logger.warning(f"YOLO load failed ({exc}) — using simulation.")
                self.use_simulation = True
        else:
            self.use_simulation = True
            logger.info("Detector: simulation mode active")

        self.subscribe(TOPIC_CAMERA, self._on_camera)
        self.subscribe(TOPIC_ODOM,   self._on_odom)

    def _on_camera(self, msg): self.latest_camera_msg = msg
    def _on_odom(self, msg):   self.latest_odom_msg   = msg

    def on_update(self, dt: float):
        if self.latest_camera_msg is None:
            return
        camera_msg = self.latest_camera_msg
        odom_msg   = self.latest_odom_msg

        t_start = time.perf_counter()

        if self.use_simulation:
            detections = self._simulate_detection(camera_msg)
        else:
            detections = self._run_yolo_inference(camera_msg)

        if odom_msg:
            detections = self._transform_to_world(detections, odom_msg)

        # NMS
        detections = self._apply_nms(detections)

        # Filter: weed class only + confidence threshold
        passed = self._filter_detections(detections)

        inference_time_ms = (time.perf_counter() - t_start) * 1000
        self.inference_times.append(inference_time_ms)
        self.detection_count += len(passed)

        detection_msg = {
            'detections':       passed,
            'count':            len(passed),
            'inference_time_ms': inference_time_ms,
            'timestamp':        camera_msg.get('timestamp', 0),
            'frame_id':         camera_msg.get('frame_id', 0),
        }
        self.publish(TOPIC_DETECTIONS, detection_msg)
        self.latest_camera_msg = None

    # ── Simulation detection (§3.3 Beta-distribution confidence) ─────────────

    def _simulate_detection(self, camera_msg: dict) -> List[Dict]:
        """
        Synthesise detections from ground-truth with Beta-sampled confidence.

        c_k = p_object × p_class_given_object × iou_score
        """
        gt_detections = camera_msg.get('detections_gt', [])
        detections = []
        cell_det_count: Dict[tuple, int] = {}

        for gt in gt_detections:
            # Determine Beta parameters by scenario
            if self.scenario == SHADOWED:
                alpha_obj, beta_obj = 5, 3      # lower mean ~0.625
            else:
                alpha_obj, beta_obj = 8, 2      # mean ~0.80

            if self.scenario == OVERLAPPING:
                alpha_iou, beta_iou = 3, 4      # lower mean ~0.43
            else:
                alpha_iou, beta_iou = 6, 2      # mean ~0.75

            p_object             = np.random.beta(alpha_obj, beta_obj)
            p_class_given_object = np.random.beta(7, 2)   # mean ~0.78
            iou_score            = np.random.beta(alpha_iou, beta_iou)
            c_k                  = p_object * p_class_given_object * iou_score

            # Track split detections (>1 per true cell)
            cell_key = (gt.get('cell_i', -1), gt.get('cell_j', -1))
            cell_det_count[cell_key] = cell_det_count.get(cell_key, 0) + 1

            gt_class = int(gt.get('class_id', 0))
            # Clamp to valid weed or generate background noise
            if gt_class not in WEED_CLASS_RANGE:
                # Simulate background / crop detection noise
                gt_class = CLASS_ID_BACKGROUND

            det = {
                'class_id':    gt_class,
                'confidence':  float(c_k),
                'bbox':        gt.get('bbox', [0.5, 0.5, 0.2, 0.2]),
                'world_x':     float(gt.get('world_x', 0)),
                'world_y':     float(gt.get('world_y', 0)),
                'cell_i':      gt.get('cell_i', 0),
                'cell_j':      gt.get('cell_j', 0),
                'scenario':    self.scenario,
                'p_object':    float(p_object),
                'p_class':     float(p_class_given_object),
                'iou':         float(iou_score),
            }
            detections.append(det)
            self.total_detections_raw += 1

        # Count cells with multiple detections
        for count in cell_det_count.values():
            if count > 1:
                self.split_detection_count += count - 1

        # Occasional false-positive noise detection
        if np.random.random() < 0.05:
            p_obj_noise = np.random.beta(1, 8)
            p_cls_noise = np.random.beta(2, 5)
            iou_noise   = np.random.beta(2, 4)
            noise_conf  = p_obj_noise * p_cls_noise * iou_noise
            detections.append({
                'class_id':   CLASS_ID_BACKGROUND,
                'confidence': float(noise_conf),
                'bbox':       [np.random.uniform(0.1, 0.9)] * 4,
                'world_x':    np.random.uniform(0, 10),
                'world_y':    np.random.uniform(0, 10),
                'cell_i':     0, 'cell_j': 0,
                'scenario':   self.scenario,
            })
            self.total_detections_raw += 1

        return detections

    # ── NMS ──────────────────────────────────────────────────────────────────

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Non-maximum suppression with IoU threshold = NMS_IOU_THRESHOLD (0.45).
        Uses cell-level proximity as IoU proxy for grid-based detections.
        """
        if len(detections) <= 1:
            return detections

        # Sort by descending confidence
        sorted_dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        keep = []
        suppressed = set()

        for i, det_i in enumerate(sorted_dets):
            if i in suppressed:
                continue
            keep.append(det_i)
            for j, det_j in enumerate(sorted_dets[i + 1:], start=i + 1):
                if j in suppressed:
                    continue
                # Cell-level IoU proxy: same or adjacent cell → suppress
                di = abs(det_i.get('cell_i', 0) - det_j.get('cell_i', 0))
                dj = abs(det_i.get('cell_j', 0) - det_j.get('cell_j', 0))
                # Bounding-box IoU approximation using bbox fields
                iou = self._bbox_iou(det_i.get('bbox', []), det_j.get('bbox', []))
                if iou >= NMS_IOU_THRESHOLD or (di == 0 and dj == 0):
                    suppressed.add(j)

        return keep

    @staticmethod
    def _bbox_iou(b1: list, b2: list) -> float:
        """Compute IoU of two [cx, cy, w, h] normalised bboxes."""
        if len(b1) < 4 or len(b2) < 4:
            return 0.0
        x1a, y1a = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
        x2a, y2a = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
        x1b, y1b = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
        x2b, y2b = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
        ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
        ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        areaa = (x2a - x1a) * (y2a - y1a)
        areab = (x2b - x1b) * (y2b - y1b)
        union = areaa + areab - inter
        return inter / union if union > 0 else 0.0

    # ── Downstream filter ─────────────────────────────────────────────────────

    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Keep only weed detections (class 0–14) with confidence > τ_c.
        Crop (15) and background (16) are silently dropped.
        """
        passed = []
        for det in detections:
            c = det.get('class_id', CLASS_ID_BACKGROUND)
            if c in WEED_CLASS_RANGE and det.get('confidence', 0) > self.conf_threshold:
                passed.append(det)
                self.confidence_log.append(det['confidence'])
                self.filtered_detections += 1
        return passed

    # ── Real YOLO inference (not used in simulation runs) ────────────────────

    def _run_yolo_inference(self, camera_msg: dict) -> List[Dict]:
        frame = camera_msg.get('raw_frame', camera_msg.get('frame'))
        if frame is None:
            return []
        if frame.dtype == np.float32 and frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                detections.append({
                    'class_id':   int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox':       box.xywhn[0].tolist(),
                    'world_x':    0.0,
                    'world_y':    0.0,
                    'cell_i':     0,
                    'cell_j':     0,
                    'scenario':   self.scenario,
                })
        return detections

    # ── Camera-to-world transform ─────────────────────────────────────────────

    def _transform_to_world(self, detections: List[Dict], odom_msg: dict) -> List[Dict]:
        robot_x = odom_msg.get('x', 0)
        robot_y = odom_msg.get('y', 0)
        robot_theta = odom_msg.get('theta', 0)
        cos_t = np.cos(robot_theta)
        sin_t = np.sin(robot_theta)
        for det in detections:
            if det['world_x'] == 0.0 and det['world_y'] == 0.0:
                bbox = det['bbox']
                cx, cy = bbox[0], bbox[1]
                cam_depth   = cy * 3.0
                cam_lateral = (cx - 0.5) * 2.0
                det['world_x'] = robot_x + cam_depth * cos_t - cam_lateral * sin_t
                det['world_y'] = robot_y + cam_depth * sin_t + cam_lateral * cos_t
        return detections

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_scenario_stats(self) -> dict:
        mean_conf = float(np.mean(self.confidence_log)) if self.confidence_log else 0.0
        return {
            "scenario":             self.scenario,
            "total_detections_raw": self.total_detections_raw,
            "filtered_detections":  self.filtered_detections,
            "mean_confidence":      round(mean_conf, 4),
            "split_detection_count": self.split_detection_count,
        }

    def get_avg_inference_time(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.mean(self.inference_times[-100:]))

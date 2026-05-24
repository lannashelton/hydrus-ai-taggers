"""
Face interrogator using ONNX models.
Models expected in ./model/:
    scrfd_2.5g_kps.onnx  – detection (multi‑scale, 5‑point landmarks)
    w600k_r50.onnx       – recognition (512‑d embeddings)
"""

import cv2
import numpy as np
import onnxruntime as ort
import insightface
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from insightface.app import FaceAnalysis


# ----------------------------------------------------------------------
#  SCRFD‑2.5G multi‑scale constants (3 scales, 2 anchors each)
# ----------------------------------------------------------------------
STRIDES = [8, 16, 32]
ANCHOR_SIZES = [
    [[4, 4], [8, 8]],       # stride 8
    [[16, 16], [32, 32]],   # stride 16
    [[32, 32], [64, 64]]    # stride 32
]
NUM_KEYPOINTS = 5

# Canonical 5‑point layout (for 112×112 input)
REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.6963],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.3655]    # right mouth
], dtype=np.float32)


def _decode_one_scale(scores, bboxes, kps, stride, anchors_wh,
                      score_thresh, img_w, img_h):
    num_anchors = len(anchors_wh)
    H = W = 640 // stride
    scores = scores.reshape(num_anchors, H, W, 1)
    bboxes = bboxes.reshape(num_anchors, H, W, 4)
    kps    = kps.reshape(num_anchors, H, W, NUM_KEYPOINTS * 2)

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid_xy = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)

    dets = []
    for a_idx, anchor in enumerate(anchors_wh):
        anchor_w, anchor_h = anchor
        score_map = scores[a_idx, ..., 0]
        bbox_map  = bboxes[a_idx]
        kps_map   = kps[a_idx]

        # Box decode (unchanged, works)
        pred_xy = 1.0 / (1.0 + np.exp(-bbox_map[..., :2]))
        pred_wh = np.exp(bbox_map[..., 2:4]) * np.array([anchor_w, anchor_h])
        center_xy = (grid_xy + pred_xy) * stride
        x1 = center_xy[..., 0] - pred_wh[..., 0] / 2
        y1 = center_xy[..., 1] - pred_wh[..., 1] / 2
        x2 = center_xy[..., 0] + pred_wh[..., 0] / 2
        y2 = center_xy[..., 1] + pred_wh[..., 1] / 2

        # Landmarks: delta * anchor_wh * stride
        lmk_xy = np.empty((H, W, 5, 2), dtype=np.float32)
        for k in range(5):
            dx = kps_map[..., k * 2]     * anchor_w * stride
            dy = kps_map[..., k * 2 + 1] * anchor_h * stride
            lmk_xy[:, :, k, 0] = center_xy[..., 0] + dx
            lmk_xy[:, :, k, 1] = center_xy[..., 1] + dy

        # Clip everything
        x1 = np.clip(x1, 0, img_w); y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w); y2 = np.clip(y2, 0, img_h)
        lmk_xy[..., 0] = np.clip(lmk_xy[..., 0], 0, img_w)
        lmk_xy[..., 1] = np.clip(lmk_xy[..., 1], 0, img_h)

        mask = score_map >= score_thresh
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            dets.append({
                'bbox': (float(x1[y, x]), float(y1[y, x]),
                         float(x2[y, x]), float(y2[y, x])),
                'score': float(score_map[y, x]),
                'landmarks': lmk_xy[y, x].astype(np.float32)
            })
    return dets


def _nms(dets: List[Dict], iou_thresh: float = 0.3,
         containment_thresh: float = 0.95) -> List[Dict]:
    """NMS with IoU, then containment suppression."""
    if len(dets) <= 1:
        return dets
    boxes = np.array([d['bbox'] for d in dets])
    scores = np.array([d['score'] for d in dets])
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    # Stage 1: standard IoU NMS
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        remain = order[1:]
        xx1 = np.maximum(x1[i], x1[remain])
        yy1 = np.maximum(y1[i], y1[remain])
        xx2 = np.minimum(x2[i], x2[remain])
        yy2 = np.minimum(y2[i], y2[remain])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[remain] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = remain[inds]

    final = [dets[i] for i in keep]
    if len(final) <= 1:
        return final

    # Stage 2: containment suppression
    final.sort(key=lambda d: d['score'], reverse=True)
    result = []
    for det in final:
        bbox_i = np.array(det['bbox'])
        area_i = (bbox_i[2] - bbox_i[0] + 1) * (bbox_i[3] - bbox_i[1] + 1)
        suppressed = False
        # Only compare to already‑kept higher‑score boxes
        for kept in result:
            bbox_j = np.array(kept['bbox'])
            area_j = (bbox_j[2] - bbox_j[0] + 1) * (bbox_j[3] - bbox_j[1] + 1)
            xi1 = max(bbox_i[0], bbox_j[0]); yi1 = max(bbox_i[1], bbox_j[1])
            xi2 = min(bbox_i[2], bbox_j[2]); yi2 = min(bbox_i[3], bbox_j[3])
            inter = max(0.0, xi2 - xi1 + 1) * max(0.0, yi2 - yi1 + 1)

            # Case A: lower‑score candidate contains a higher‑score box → suppress it
            if inter / area_j > containment_thresh:
                suppressed = True
                break
            # Case B: candidate is fully inside a higher‑score box
            if inter / area_i > containment_thresh:
                suppressed = True
                break
        if not suppressed:
            result.append(det)
    return result


def _decode_all(outputs, img_w, img_h, score_thresh=0.6):
    """Decode all scales and apply NMS."""
    scores = outputs[0:3]
    bboxes = outputs[3:6]
    kps    = outputs[6:9]

    all_dets = []
    for si, stride in enumerate(STRIDES):
        anchors = ANCHOR_SIZES[si]
        dets = _decode_one_scale(
            scores[si].ravel(), bboxes[si].ravel(), kps[si].ravel(),
            stride, anchors, score_thresh, img_w, img_h
        )
        all_dets.extend(dets)

    return _nms(all_dets, iou_thresh=0.3, containment_thresh=0.95)


def _is_valid_face(bbox, img_w, img_h):
    """Reject boxes that are too small, too large, or have extreme aspect ratios."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w < 20 or h < 20:
        return False
    if w > img_w * 0.9 or h > img_h * 0.9:
        return False
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > 3.0:
        return False
    return True


def _align_face(img_bgr: np.ndarray, det: Dict,
                target_size=(112, 112)) -> np.ndarray:
    """Warp face to canonical position using 5 landmarks."""
    landmarks = det['landmarks'].astype(np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(landmarks, REFERENCE_POINTS)
    if matrix is None:
        x1, y1, x2, y2 = det['bbox']
        crop = img_bgr[max(0, int(y1)): int(y2), max(0, int(x1)): int(x2)]
        return cv2.resize(crop, target_size)
    return cv2.warpAffine(img_bgr, matrix, target_size, flags=cv2.INTER_LINEAR)


# In FaceInterrogator.__init__ (inside interrogate_faces.py)
class FaceInterrogator:
    def __init__(self, det_model, rec_model, device="CPU"):
        if device.upper() == "NPU":
            providers = [
                ('OpenVINOExecutionProvider', {
                    'device_type': 'NPU',
                    'precision': 'FP16'
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_thresh=0.5)

    def detect_faces(self, image_bgr, conf=0.6):
        faces = self.app.get(image_bgr, max_num=0)   # max_num=0 means unlimited
        result = []
        for face in faces:
            if face.det_score < conf:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            result.append({
                'bbox': (x1, y1, x2, y2),
                'score': float(face.det_score),
                'landmarks': face.kps.astype(np.float32) if face.kps is not None else None,
                'embedding': face.normed_embedding   # already L2‑normalized
            })
        return result

    def get_embedding(self, image_bgr, det):
        # Embedding already comes with the detection; we just return it.
        return det.get('embedding', None)

"""Pre/post-processing helpers for the RKNN YOLO11 detector.

These mirror the preprocessing that Ultralytics applied when the original
``best.pt`` model was served, so that the RKNN model sees exactly the same
input distribution and the detection boxes line up with ``regions.csv``.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

PAD_COLOR = (114, 114, 114)
"""Letterbox padding color, identical to the Ultralytics default."""


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = PAD_COLOR,
) -> tuple[np.ndarray, float, int, int]:
    """Resize ``img`` to ``new_shape`` while preserving aspect ratio.

    The image is padded with ``color``. This reproduces Ultralytics' ``LetterBox``
    (``scaleup=True``, ``center=True``, ``INTER_LINEAR``, pad ``114``).

    Returns a tuple of ``(padded_img, gain, pad_x, pad_y)`` where ``gain`` is the
    uniform scale factor and ``(pad_x, pad_y)`` are the left/top padding offsets
    used to map coordinates back to the original image.
    """
    shape = img.shape[:2]  # (height, width)
    gain = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (round(shape[1] * gain), round(shape[0] * gain))  # (width, height)

    if (shape[1], shape[0]) != new_unpad:
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    pad_x = round(dw / 2 - 0.1)  # left
    pad_y = round(dh / 2 - 0.1)  # top
    top, bottom = pad_y, round(dh / 2 + 0.1)
    left, right = pad_x, round(dw / 2 + 0.1)

    img = cv.copyMakeBorder(
        img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color
    )
    return img, gain, pad_x, pad_y


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Suppress overlapping boxes, keeping the highest scoring ones.

    ``boxes`` is an ``(N, 4)`` array of ``xyxy`` coordinates and ``scores`` is an
    ``(N,)`` array. Returns the indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def decode_output(
    output: np.ndarray,
    orig_shape: tuple[int, int],
    gain: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    """Decode a standard Ultralytics YOLO11 export output into ``xyxy`` boxes.

    The model exports a single tensor of shape ``(1, 4 + nc, num_anchors)`` whose
    first four channels are the decoded boxes and the remaining channels are
    sigmoid-activated class scores. For the Ultralytics YOLO11 head (default
    ``xywh`` export), those four channels are ``[cx, cy, w, h]`` in letterboxed
    pixel space, so they are converted to ``xyxy`` here.

    Returns an ``(M, 4)`` array of ``xyxy`` boxes rescaled to the original image
    coordinates, where ``M`` is the number of detections of the packed class that
    survive the confidence threshold and NMS.
    """
    from .config import CONF_THRESHOLD, NMS_THRESHOLD, PACKED_CLASS_ID

    pred = output[0]  # (4 + nc, num_anchors)
    # # [cx, cy, w, h] -> [x1, y1, x2, y2]
    cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
    boxes = np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), axis=1)
    class_scores = pred[4:].T  # (num_anchors, nc)
    scores = class_scores[:, PACKED_CLASS_ID]

    mask = scores >= CONF_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    keep = non_max_suppression(boxes, scores, NMS_THRESHOLD)
    boxes = boxes[keep]

    # Undo letterboxing: subtract padding then divide by the scale factor.
    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, 0] -= pad_x
    boxes[:, 2] -= pad_x
    boxes[:, 1] -= pad_y
    boxes[:, 3] -= pad_y
    boxes /= gain

    h, w = orig_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

    return boxes

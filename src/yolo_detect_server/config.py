ROI_RATIO = 0.4
"""
Search region ratio of packing locations.

if the center point of detection is in this region, then the object is
considered to be at the corresponding location. The region is placed at the
center of the region of the location, with width and height scaled using this
ratio.
"""

RKNN_MODEL_PATH = "weights/best.rknn"
"""Path to the RKNN model used for inference."""

TARGET_PLATFORM = "rk3588"
"""RKNPU target platform the model was converted for."""

DEVICE_ID: str | None = None
"""Device id of the RKNPU to run inference on.

   ``None`` lets rknn-toolkit2 auto-select the connected board (usb/adb or
   network). Set it to an explicit adb serial number or ntb device id when more
   than one board is connected.
"""

IMG_SIZE = (640, 640)
"""(height, width) the model was trained/exported with.

   The RKNN model has this size baked in at conversion time, so input frames
   are letterboxed to exactly this shape before inference.
"""

CONF_THRESHOLD = 0.6
"""Minimum confidence of a detection to be considered valid."""

NMS_THRESHOLD = 0.45
"""IoU threshold used by non-maximum suppression."""

PACKED_CLASS_ID = 0
"""Class index that represents a packed glass (see ``weights`` model names)."""

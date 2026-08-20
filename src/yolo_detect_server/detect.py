import numpy as np
import polars as pl
from rknn.api import RKNN

from .config import DEVICE_ID, IMG_SIZE, RKNN_MODEL_PATH, TARGET_PLATFORM
from .postprocess import decode_output, letterbox
from .util import xywh_to_xyxy


def _init_model() -> RKNN:
    """Load the RKNN model and initialise the RKNPU runtime."""
    rknn = RKNN()

    if (ret := rknn.load_rknn(RKNN_MODEL_PATH)) != 0:
        raise RuntimeError(f"load_rknn failed with error code {ret}")

    if (ret := rknn.init_runtime(target=TARGET_PLATFORM, device_id=DEVICE_ID)) != 0:
        raise RuntimeError(f"init_runtime failed with error code {ret}")

    return rknn


model = _init_model()

df = pl.read_csv("regions.csv")
REGIONS = [xywh_to_xyxy(row) for row in df.iter_rows(named=True)]


def find_packed(img: np.ndarray) -> list[str]:
    orig_h, orig_w = img.shape[:2]
    letterboxed, gain, pad_x, pad_y = letterbox(img, new_shape=IMG_SIZE)
    rgb = letterboxed[..., ::-1]  # BGR -> RGB, as Ultralytics did

    outputs = model.inference(inputs=[rgb])
    boxes = decode_output(
        outputs[0],
        orig_shape=(orig_h, orig_w),
        gain=gain,
        pad_x=pad_x,
        pad_y=pad_y,
    )

    packed = []
    # TODO: 理论上存在误检导致相同库位中存在多个结果的可能性，后续可以考虑进行优化
    for box in boxes:
        x = (box[0] + box[2]) / 2 / orig_w
        y = (box[1] + box[3]) / 2 / orig_h
        for region in REGIONS:
            if region.contains(x, y):
                packed.append({"name": region.name, "priority": region.priority})
                break
    packed.sort(key=lambda region: region["priority"])

    return [region["name"] for region in packed]

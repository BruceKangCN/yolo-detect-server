import numpy as np
import polars as pl
from ultralytics import YOLO

from .config import MODEL_PATH
from .util import xywh_to_xyxy

model = YOLO(MODEL_PATH)

df = pl.read_csv("regions.csv")
REGIONS = [xywh_to_xyxy(row) for row in df.iter_rows(named=True)]


def find_packed(img: np.ndarray) -> list[str]:
    result = model.predict(img, conf=0.6)[0]  # type: ignore

    packed = []
    # TODO: 理论上存在误检导致相同库位中存在多个结果的可能性，后续可以考虑进行优化
    for box in result.boxes:  # type: ignore
        if box.cls.cpu().numpy()[0] != 0:  # type: ignore
            continue
        x, y, _, _ = box.xywhn.cpu().numpy()[0]  # type: ignore
        for region in REGIONS:
            if region.contains(x, y):
                packed.append({"name": region.name, "priority": region.priority})
                break
    sorted(packed, key=lambda region: region["priority"])

    return [region["name"] for region in packed]

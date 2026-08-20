import numpy as np
import polars as pl
from ultralytics import YOLO

from .util import xywh_to_xyxy


class Detector:
    def __init__(self, path):
        self.model = YOLO(path)
        df = pl.read_csv("regions.csv")
        self.regions = [xywh_to_xyxy(row) for row in df.iter_rows(named=True)]

    def find_packed(self, img: np.ndarray) -> list[str]:
        result = self.model.predict(img, conf=0.6)[0]  # type: ignore

        packed = []
        # TODO: 理论上存在误检导致相同库位中存在多个结果的可能性，后续可以考虑进行优化
        for box in result.boxes:  # type: ignore
            if box.cls.cpu().numpy()[0] != 0:  # type: ignore
                continue
            x, y, _, _ = box.xywhn.cpu().numpy()[0]  # type: ignore
            for region in self.regions:
                if region.contains(x, y):
                    packed.append({"name": region.name, "priority": region.priority})
                    break
        packed.sort(key=lambda region: region["priority"])

        return [region["name"] for region in packed]

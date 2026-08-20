from pydantic import BaseModel

from .config import ROI_RATIO


class Region(BaseModel):
    name: str
    priority: int = 0
    t: float  # top
    l: float  # left
    b: float  # bottom
    r: float  # right

    def contains(self, x: float, y: float) -> bool:
        if x < self.l or x > self.r:
            return False
        if y < self.t or y > self.b:  # noqa: SIM103
            return False
        return True


class Detection(BaseModel):
    """detection result"""

    code: int
    """
    status code of detection.
    - 0: success
    - 1: failure
    """

    packed: list[str] | None = None
    """
    A list of all the places with a packed glass, sorted by place priorities.
    This field only exists if the detection succeed.

    # Value Example

    ```python
    ["A11", "B12"]
    ```
    """


def xywh_to_xyxy(row: dict) -> Region:
    name = row["name"]
    priority = row["priority"]
    l = row["x"] - row["w"] / 2 * ROI_RATIO
    r = row["x"] + row["w"] / 2 * ROI_RATIO
    t = row["y"] - row["h"] / 2 * ROI_RATIO
    b = row["y"] + row["h"] / 2 * ROI_RATIO
    return Region(name=name, priority=priority, t=t, l=l, b=b, r=r)

from pydantic import BaseModel

from .config import ROI_RATIO


class Region(BaseModel):
    name: str
    t: float # top
    l: float # left
    b: float # bottom
    r: float # right

    def contains(self, x: float, y: float) -> bool:
        if x < self.l or x > self.r:
            return False
        if y < self.t or y > self.b:
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


def xywh_to_xyxy(row: tuple[str, float, float, float, float]) -> Region:
    name = row[0]
    l = row[1] - row[3] / 2 * ROI_RATIO
    r = row[1] + row[3] / 2 * ROI_RATIO
    t = row[2] - row[4] / 2 * ROI_RATIO
    b = row[2] + row[4] / 2 * ROI_RATIO
    return Region(name=name, t=t, l=l, b=b, r=r)

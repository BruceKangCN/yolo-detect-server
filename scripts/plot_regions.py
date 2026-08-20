# %%
import builtins
from pathlib import Path

import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl

mpl.rc("figure", figsize=[9.6, 7.2])
if getattr(builtins, "__IPYTHON__", False):
    mpl.use("widget")
else:
    mpl.use("QtAgg")


# %%
BG_IMG_PATH = "images/0002.jpg"

csv_path = Path(__file__).parent.parent / "regions.csv"


def xywhn_to_xyxy(
    x: float,
    y: float,
    w: float,
    h: float,
    img_size: tuple[int, int] = (1600, 1200),
    ratio: float = 0.4,
) -> tuple[int, int, int, int]:
    l = int((x - w / 2 * ratio) * img_size[0])
    t = int((y - h / 2 * ratio) * img_size[1])
    r = int((x + w / 2 * ratio) * img_size[0])
    b = int((y + h / 2 * ratio) * img_size[1])
    return l, t, r, b


# %%
def main():
    df = pl.read_csv(csv_path)
    img = cv.imread(BG_IMG_PATH, cv.IMREAD_COLOR_RGB)
    h, w = img.shape[:2]
    img_size = (w, h)

    for row in df.iter_rows(named=True):
        l, t, r, b = xywhn_to_xyxy(
            row["x"],
            row["y"],
            row["w"],
            row["h"],
            img_size=img_size,
            ratio=0.4,
        )
        cv.rectangle(img, (l, t), (r, b), color=(102, 255, 204), thickness=3)
        cv.putText(
            img,
            text=row["name"],
            org=(l, t),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
        )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    plt.show()


# %%
if __name__ == "__main__":
    main()

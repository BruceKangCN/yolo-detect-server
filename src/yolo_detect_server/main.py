import uvicorn
from fastapi import FastAPI

from .detect import find_packed
from .util import Detection


app = FastAPI()

@app.get("/", summary="detect pack place status")
async def detect() -> Detection:
    """
    Detect packs using YOLO model, and find out all the places with packed
    glasses. A detection status code will be returned, which can be used to
    check whether the detection succeed or not. A list of place names sorted by
    their place priorities will also be returned on success. No list will be
    returned on failure.
    """

    try:
        import numpy as np
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) # TODO
        packed = find_packed(img)
        return Detection(code=0, packed=packed)
    except Exception as ex:
        print(ex)
        return Detection(code=1)


@app.get("/test", summary="test endpoint which returns a selected place")
async def selectdetection(name: str | None = None) -> Detection:
    if name is None:
        return Detection(code=1)
    return Detection(code=0, packed=[name])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

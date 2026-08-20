from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from yolo_detect_server.camera import Camera
from yolo_detect_server.config import MODEL_PATH
from yolo_detect_server.detect import Detector
from yolo_detect_server.util import Detection


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Detector(MODEL_PATH)
    app.state.camera = Camera()

    yield


async def get_model(requst: Request) -> Detector:
    return requst.app.state.model


async def get_camera(request: Request) -> Camera:
    return request.app.state.camera


ModelDep = Annotated[Detector, Depends(get_model)]
CameraDep = Annotated[Camera, Depends(get_camera)]

app = FastAPI(lifespan=lifespan)


@app.get("/", summary="detect pack place status")
async def detect(model: ModelDep, camera: CameraDep) -> Detection:
    """
    Detect packs using YOLO model, and find out all the places with packed
    glasses. A detection status code will be returned, which can be used to
    check whether the detection succeed or not. A list of place names sorted by
    their place priorities will also be returned on success. No list will be
    returned on failure.
    """

    try:
        img = camera.get_frame()
        if img is None:
            return Detection(code=1)
        packed = model.find_packed(img)
        return Detection(code=0, packed=packed)
    except RuntimeError as ex:
        print(ex)
        return Detection(code=1)


@app.get("/test", summary="test endpoint which returns a selected place")
async def select_region(name: str | None = None) -> Detection:
    if name is None:
        return Detection(code=1)
    return Detection(code=0, packed=[name])

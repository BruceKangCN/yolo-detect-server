import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from yolo_detect_server.camera import Camera
from yolo_detect_server.config import MODEL_PATH
from yolo_detect_server.detect import Detector
from yolo_detect_server.frame_receiver import FrameReceiver
from yolo_detect_server.util import Detection


@asynccontextmanager
async def lifespan(app: FastAPI):
    stream_url = os.environ.get("STREAM_URL", default="rtsp://localhost")

    app.state.model = Detector(MODEL_PATH)
    app.state.camera = Camera()
    app.state.frame_receiver = FrameReceiver(stream_url)

    yield


async def get_model(request: Request) -> Detector:
    return request.app.state.model


async def get_camera(request: Request) -> Camera:
    return request.app.state.camera


async def get_frame_receiver(request: Request) -> FrameReceiver:
    return request.app.state.frame_receiver


ModelDep = Annotated[Detector, Depends(get_model)]
CameraDep = Annotated[Camera, Depends(get_camera)]
FrameReceiverDep = Annotated[FrameReceiver, Depends(get_frame_receiver)]

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def detect_from_fixed(model: ModelDep, camera: CameraDep) -> Detection:
    """Detect pack place status.

    Detect packs using frame from fixed camera with YOLO model, and find out all
    the places with packed glasses. A detection status code will be returned,
    which can be used to check whether the detection succeed or not. A list of
    place names sorted by their place priorities will also be returned on
    success. No list will be returned on failure.
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


@app.get("/vihecle")
async def detect_from_vihecle(model: ModelDep, frame_receiver: FrameReceiverDep):
    """Detect packs using camera on vihecles (e.g. AGVs).

    Detect packs using frame from camera on vihecles with YOLO model, and find
    out locations of packs.

    TODO: maybe we should use RGBD camera to get precise relative location of
    packs from the vihecle to improve detection of locations.
    """
    img = frame_receiver.get_frame()
    if img is None:
        return Detection(code=1)
    packed = model.find_packed(img)
    return Detection(code=0, packed=packed)


@app.get("/test", summary="test endpoint which returns a selected place")
async def select_region(name: str | None = None) -> Detection:
    if name is None:
        return Detection(code=1)
    return Detection(code=0, packed=[name])

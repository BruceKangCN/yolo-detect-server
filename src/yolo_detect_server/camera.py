import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SDK_PATH = PROJECT_ROOT / "ScepterSDK" / "MultilanguageSDK" / "Python"
sys.path.insert(0, os.path.abspath(SDK_PATH))

import time
from ctypes import c_uint16

import numpy as np

from API.ScepterDS_api import ScepterTofCam, ScConnectStatus, ScFrameType # type: ignore


camera = ScepterTofCam()
camera_count = camera.scGetDeviceCount(3000)
print(f"[scGetDeviceCount] success, ScStatus(0), device count: {camera_count}")
if camera_count <= 0:
    print("[scGetDeviceCount] No device found after scanning for 3000ms. Make sure your device is connected.")
    exit(1)

ret, device_infolist = camera.scGetDeviceInfoList(camera_count)
if ret != 0:
    print("[scGetDeviceInfoList] fail, ScStatus({})".format(ret))
    exit(1)

device_info = device_infolist[0]
print(
    "[scGetDeviceInfoList] success, ScStatus({}). Display the first deviceInfo, <serialNumber>: {} , <ip>: {} , <status>: {}".format(
        ret,
        str(device_info.serialNumber.decode()),
        str(device_info.ip.decode()),
        str(device_info.status),
    )
)
if ScConnectStatus.SC_CONNECTABLE.value != device_info.status:
    print(
        " The first device [status]: {} does not support connection.".format(
            str(device_info.status)
        )
    )
    exit(1)

if (ret := camera.scOpenDeviceBySN(device_info.serialNumber)) != 0:
    print(f"[scOpenDeviceBySN] fail ScStatus({ret}).")
    exit(1)
print("[scOpenDeviceBySN] success ScStatus({}).".format(str(ret)))

if (ret := camera.scStartStream()) != 0:
    print("[scStartStream] fail ScStatus({}).".format(str(ret)))
    exit(1)

print("[scStartStream] success ScStatus({}).".format(str(ret)))
# Wait for the device to upload image data.
time.sleep(1)

colorSlope = c_uint16(7495)

def get_frame() -> np.ndarray | None:
    ret, frame_ready = camera.scGetFrameReady(c_uint16(1200))
    if ret != 0:
        print(f"[scGetFrameReady] failed: ScStatus({ret}).")
        return None

    if not frame_ready.color:
        print("missing color frame")
        return None

    ret, frame = camera.scGetFrame(ScFrameType.SC_COLOR_FRAME)
    if ret != 0:
        print(f"[scGetFrame] get color frame failed: ScStatus({ret}).")

    img = np.ctypeslib.as_array(frame.pFrameData, (1, frame.width * frame.height * 3))
    img = img.view(np.uint8)
    img = img.reshape((frame.height, frame.width, 3))

    return img

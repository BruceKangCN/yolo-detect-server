import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SDK_PATH = PROJECT_ROOT / "ScepterSDK" / "MultilanguageSDK" / "Python"
sys.path.insert(0, os.path.abspath(SDK_PATH))

import time
from ctypes import c_uint16

import numpy as np
from API.ScepterDS_api import (  # type: ignore
    ScConnectStatus,
    ScepterTofCam,
    ScFrameType,
)


class Camera:
    def __init__(self) -> None:
        self.camera = ScepterTofCam()

        camera_count = self.camera.scGetDeviceCount(3000)
        print(f"[scGetDeviceCount] success, ScStatus(0), device count: {camera_count}")
        if camera_count <= 0:
            print(
                "[scGetDeviceCount] No device found after scanning for 3000ms. Make sure your device is connected.",
            )
            raise RuntimeError("No camera device found")

        ret, device_infolist = self.camera.scGetDeviceInfoList(camera_count)
        if ret != 0:
            print(f"[scGetDeviceInfoList] fail, ScStatus({ret})")
            raise RuntimeError("Failed to get device info list")

        device_info = device_infolist[0]
        print(
            " ".join(
                [
                    f"[scGetDeviceInfoList] success, ScStatus({ret}).",
                    "Display the first device info,",
                    f"<serialNumber>: {device_info.serialNumber.decode()}",
                    f"<ip>: {device_info.ip.decode()}",
                    f"<status>: {device_info.status}",
                ]
            ),
        )
        if device_info.status != ScConnectStatus.SC_CONNECTABLE.value:
            print(
                f" The first device [status]: {device_info.status} does not support connection."
            )
            raise RuntimeError("Cannot connect to device")

        if (ret := self.camera.scOpenDeviceBySN(device_info.serialNumber)) != 0:
            print(f"[scOpenDeviceBySN] fail ScStatus({ret}).")
            raise RuntimeError("Cannot open device")
        print(f"[scOpenDeviceBySN] success ScStatus({ret}).")

        if (ret := self.camera.scStartStream()) != 0:
            print(f"[scStartStream] fail ScStatus({ret}).")
            raise RuntimeError("Cannot start stream")

        print(f"[scStartStream] success ScStatus({ret}).")
        # Wait for the device to upload image data.
        time.sleep(1)

    def __del__(self):
        ret = self.camera.scStopStream()
        if ret != 0:
            print(f"[scStopStream] Warning: failed to stop stream ({ret})")
        ret = self.camera.scCloseDevice()
        if ret != 0:
            print(f"[scCloseDevice] Warning: failed to coles device ({ret})")

    def get_frame(self) -> np.ndarray | None:
        ret, frame_ready = self.camera.scGetFrameReady(c_uint16(1200))
        if ret != 0:
            print(f"[scGetFrameReady] failed: ScStatus({ret}).")
            return None

        if not frame_ready.color:
            print("missing color frame")
            return None

        ret, frame = self.camera.scGetFrame(ScFrameType.SC_COLOR_FRAME)
        if ret != 0:
            print(f"[scGetFrame] get color frame failed: ScStatus({ret}).")

        img = np.ctypeslib.as_array(
            frame.pFrameData, (1, frame.width * frame.height * 3)
        )
        img = img.view(np.uint8)
        img = img.reshape((frame.height, frame.width, 3))

        return img

import cv2 as cv
import numpy as np


class FrameReceiver:
    def __init__(self, url: str) -> None:
        self.cap = cv.VideoCapture(url)
        self.max_flush = 50

    def __del__(self):
        self.cap.release()

    def get_frame(self) -> np.ndarray:
        # Flush frame buffer to get the latest frame. Limit max flush times to
        # prevent infinite flush if stream fps is higher than process fps.
        for _ in range(self.max_flush):
            # grab until no frame is in the buffer
            if not self.cap.grab():
                break

        ret, frame = self.cap.retrieve()
        if not ret:
            raise RuntimeError("Failed to retrieve frame")

        return frame

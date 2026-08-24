import os

import cv2 as cv
from dotenv import load_dotenv


def main():
    load_dotenv()

    stream_url = os.environ.get("STREAM_URL", default="rtsp://localhost:8554")
    print(f"stream URL: {stream_url}")
    cap = cv.VideoCapture(stream_url)

    ret = True
    while ret:
        ret, frame = cap.read()
        cv.imshow("Viewer", frame)

        if cv.waitKey(1) & 0xFF == 27:  # use ESC to close
            break

    cap.release()


if __name__ == "__main__":
    main()

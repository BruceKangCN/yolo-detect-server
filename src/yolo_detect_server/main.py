import uvicorn
from dotenv import load_dotenv

from yolo_detect_server.app import app

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)

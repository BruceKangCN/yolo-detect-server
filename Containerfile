FROM ghcr.io/astral-sh/uv:0.12.6-python3.12-trixie

WORKDIR /app

VOLUME [ "/app/weights" ]

COPY \
    README.md \
    LICENSE \
    pyproject.toml \
    .python-version \
    uv.lock \
    .env \
    regions.csv \
    ./
COPY src src
COPY ScepterSDK ScepterSDK

RUN uv sync --frozen --no-dev

CMD [ "uv", "run", "src/yolo_detect_server/main.py" ]

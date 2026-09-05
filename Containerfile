FROM ghcr.io/astral-sh/uv:0.12.6-python3.12-trixie

WORKDIR /app

VOLUME [ "/app/weights" ]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

CMD [ "uv", "run", "--no-dev", "src/yolo_detect_server/main.py" ]

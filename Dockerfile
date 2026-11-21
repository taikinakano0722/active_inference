FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MUJOCO_GL=egl \
    PATH="/opt/venv/bin:$PATH"

# DreamerV3 は Python 3.11+。MuJoCo の EGL 描画に GL ライブラリが必要。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglew2.2 \
    libglfw3 \
    libglib2.0-0 \
    libosmesa6 \
    patchelf \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv --upgrade-deps \
    && pip install -U pip setuptools wheel

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && AutoROM --accept-license

# ソースは docker-compose が /workspace にマウントする。
CMD ["sleep", "infinity"]

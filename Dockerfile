FROM python:3.9-slim

# 系统依赖（Debian 13 包名）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 libxext6 libxrender1 libgomp1 \
    wget curl git unzip \
    && rm -rf /var/lib/apt/lists/*

# 换清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 核心依赖
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    onnx==1.19.1 \
    onnxruntime==1.17.1 \
    opencv-python-headless==4.8.1.78 \
    numpy==1.24.3 \
    pillow \
    matplotlib \
    ultralytics \
    scikit-learn \
    requests \
    tqdm \
    h5py \
    pandas \
    seaborn

WORKDIR /workspace
CMD ["/bin/bash"]

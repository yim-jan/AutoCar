FROM python:3.10-slim

# 安装依赖
RUN pip install --no-cache-dir \
    onnxruntime \
    opencv-python-headless \
    numpy \
    Pillow

# 创建工作目录
WORKDIR /workspace

# 默认命令
CMD ["python3"]

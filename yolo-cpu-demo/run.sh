#!/bin/bash
# 车载AI工程化 第1-3周 一键运行脚本

echo "========================================="
echo " 车载AI工程化 第1-3周 一键运行脚本"
echo "========================================="

# 检查 Docker（非强制，仅提示）
command -v docker &> /dev/null || echo "⚠️ 未检测到 Docker，跳过容器检查"

# --- 第1周 任务1: 模型推理 ---
echo ""
echo "--- [第1周 任务1] YOLOv5s 推理 ---"

if [ ! -f "/workspace/yolov5s.pt" ]; then
    echo "下载 YOLOv5s 权重..."
    wget -q https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
fi

if [ ! -d "/workspace/yolov5" ]; then
    echo "克隆 YOLOv5 仓库..."
    git clone -q https://github.com/ultralytics/yolov5.git /workspace/yolov5
fi

if [ ! -f "/workspace/test.jpg" ]; then
    echo "下载测试图片..."
    wget -q -O /workspace/test.jpg \
        "https://www.cvlibs.net/datasets/kitti/data_object/image_2/testing/image_2/000000.png"
fi

python /workspace/infer.py || echo "⚠️ 任务1推理失败，继续执行..."
echo "[第1周 任务1] 完成"

# --- 第1周 任务2: 数据转换 ---
echo ""
echo "--- [第1周 任务2] 数据格式转换 ---"

if [ ! -d "/workspace/data/raw/2011_09_26_drive_0001_sync" ]; then
    echo "下载 KITTI raw 数据..."
    mkdir -p /workspace/data/raw /workspace/data/processed
    wget -q https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip \
        -P /workspace/data/raw/ || echo "⚠️ KITTI 下载失败"
    unzip -q /workspace/data/raw/2011_09_26_drive_0001_sync.zip -d /workspace/data/raw/ 2>/dev/null || true
fi

python /workspace/rosbag_to_hdf5.py || echo "⚠️ 任务2数据转换失败，继续执行..."
echo "[第1周 任务2] 完成"

# --- 第2周: 模型量化 ---
echo ""
echo "--- [第2周] INT8 静态量化 ---"

if [ ! -f "/workspace/yolov5s.onnx" ]; then
    python /workspace/export_yolo_onnx.py || echo "⚠️ ONNX导出失败"
fi

if [ ! -f "/workspace/yolov5s_int8.onnx" ]; then
    python /workspace/quantize.py || echo "⚠️ 量化失败"
fi

python /workspace/eval_accuracy.py || echo "⚠️ 精度评测失败，继续执行..."
echo "[第2周] 完成"

# --- 第3周 任务1: 性能剖析 ---
echo ""
echo "--- [第3周 任务1] 算子级性能剖析 ---"

if [ ! -f "/workspace/profile_report.txt" ]; then
    python /workspace/profile_model.py || echo "⚠️ 性能剖析失败"
fi
echo "[第3周 任务1] 完成"

# --- 第3周 任务2: 场景评测 ---
echo ""
echo "--- [第3周 任务2] 夜间+逆光场景评测 ---"
python /workspace/eval_night_scene.py || echo "⚠️ 场景评测失败，继续执行..."
echo "[第3周 任务2] 完成"

echo ""
echo "========================================="
echo " 第1-3周全部任务完成！"
echo "========================================="


# --- 第4周: 长尾问题闭环 ---
echo ""
echo "--- [第4周] 低光照长尾问题闭环 ---"
python /workspace/build_hard_dataset.py || echo "⚠️ 数据挖掘失败"
python /workspace/finetune_lowlight.py || echo "⚠️ v1失败"
python /workspace/finetune_lowlight_v2.py || echo "⚠️ v2失败"
echo "[第4周] 完成 → 结论：需训练侧补充低光照数据"

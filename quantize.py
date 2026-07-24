"""
第2周：模型量化脚本
对 YOLOv5s ONNX 做 INT8 静态量化，对比量化前后精度和速度
"""
import os
import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import onnx
from preprocess import preprocess_image

# 路径配置
MODEL_FP32 = "/workspace/yolov5s.onnx"
MODEL_INT8 = "/workspace/yolov5s_int8.onnx"
CALIB_DIR = "/workspace/data/coco_calib"


class CalibrationDataLoader(CalibrationDataReader):
    """校准数据加载器：遍历calib图片，输出模型输入格式的张量"""
    
    def __init__(self, calib_dir, num_calib=200):
        self.calib_dir = calib_dir
        self.files = sorted(os.listdir(calib_dir))[:num_calib]
        self.index = 0
        self.input_name = "images"

    def get_next(self):
        if self.index >= len(self.files):
            return None
        
        img_path = os.path.join(self.calib_dir, self.files[self.index])
        self.index += 1
        
        try:
            tensor, _ = preprocess_image(img_path)
            return {self.input_name: tensor}
        except Exception as e:
            print(f"[校准警告] {img_path}: {e}")
            return self.get_next()  # 跳过损坏图片，读下一张


def benchmark(model_path, num_runs=100):
    """
    测试模型推理速度
    返回: 平均推理时间(ms), FPS
    """
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # 构造随机输入
    dummy = np.random.randn(*input_shape).astype(np.float32)
    
    # 预热
    for _ in range(10):
        session.run(None, {input_name: dummy})
    
    # 计时
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy})
    elapsed = time.time() - start
    
    avg_ms = (elapsed / num_runs) * 1000
    fps = 1000 / avg_ms
    return avg_ms, fps


def get_model_size(model_path):
    """返回模型文件大小(MB)"""
    return os.path.getsize(model_path) / (1024 * 1024)


def main():
    print("=" * 50)
    print("第2周：YOLOv5s INT8 静态量化")
    print("=" * 50)
    
    # === 1. FP32基准测试 ===
    print("\n[1/4] FP32基准测试...")
    fp32_size = get_model_size(MODEL_FP32)
    fp32_time, fp32_fps = benchmark(MODEL_FP32)
    print(f"  模型大小: {fp32_size:.1f} MB")
    print(f"  推理时间: {fp32_time:.2f} ms")
    print(f"  FPS: {fp32_fps:.1f}")
    
    # === 2. 执行静态量化 ===
    print("\n[2/4] 执行INT8静态量化...")
    
    # 先检查ONNX模型是否有效
    onnx.checker.check_model(MODEL_FP32)
    print("  ONNX模型检查通过")
    
    # 量化
    quantize_static(
        model_input=MODEL_FP32,
        model_output=MODEL_INT8,
        calibration_data_reader=CalibrationDataLoader(CALIB_DIR, num_calib=200),
        quant_format=QuantFormat.QOperator,  # 使用QOperator格式，CPU兼容性好
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    print(f"  INT8模型已保存: {MODEL_INT8}")
    
    # === 3. INT8基准测试 ===
    print("\n[3/4] INT8基准测试...")
    int8_size = get_model_size(MODEL_INT8)
    int8_time, int8_fps = benchmark(MODEL_INT8)
    print(f"  模型大小: {int8_size:.1f} MB")
    print(f"  推理时间: {int8_time:.2f} ms")
    print(f"  FPS: {int8_fps:.1f}")
    
    # === 4. 对比报告 ===
    print("\n[4/4] 量化对比报告")
    print("=" * 50)
    print(f"{'指标':<15} {'FP32':<15} {'INT8':<15} {'变化':<15}")
    print("-" * 50)
    print(f"{'模型大小(MB)':<15} {fp32_size:<15.1f} {int8_size:<15.1f} {int8_size/fp32_size-1:<15.1%}")
    print(f"{'推理时间(ms)':<15} {fp32_time:<15.2f} {int8_time:<15.2f} {int8_time/fp32_time-1:<15.1%}")
    print(f"{'FPS':<15} {fp32_fps:<15.1f} {int8_fps:<15.1f} {int8_fps/fp32_fps-1:<15.1%}")
    print("=" * 50)
    
    # 保存报告到文件
    with open("/workspace/quantization_report.txt", 'w') as f:
        f.write(f"FP32 模型大小: {fp32_size:.1f} MB\n")
        f.write(f"INT8 模型大小: {int8_size:.1f} MB\n")
        f.write(f"FP32 推理时间: {fp32_time:.2f} ms, FPS: {fp32_fps:.1f}\n")
        f.write(f"INT8 推理时间: {int8_time:.2f} ms, FPS: {int8_fps:.1f}\n")
        f.write(f"速度提升: {fp32_fps/int8_fps:.2f}x\n")
        f.write(f"模型压缩: {(1-int8_size/fp32_size)*100:.1f}%\n")
    print("\n报告已保存到 quantization_report.txt")


if __name__ == "__main__":
    main()
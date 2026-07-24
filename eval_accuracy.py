"""
第2周：量化精度评测
对比 FP32 和 INT8 模型在相同图片上的检测结果差异
"""
import os
import numpy as np
import onnxruntime as ort
from preprocess import preprocess_image

MODEL_FP32 = "/workspace/yolov5s.onnx"
MODEL_INT8 = "/workspace/yolov5s_int8.onnx"
TEST_DIR = "/workspace/data/coco_calib"
OUTPUT_REPORT = "/workspace/accuracy_report.txt"


def run_inference(session, input_tensor):
    """跑一次推理，返回原始输出"""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})
    return output[0]


def compare_outputs(fp32_out, int8_out):
    """
    对比两个模型的输出差异
    返回: 最大差异, 平均差异, 余弦相似度
    """
    fp32_flat = fp32_out.flatten()
    int8_flat = int8_out.flatten()
    
    diff = np.abs(fp32_flat - int8_flat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # 余弦相似度（越接近1越好）
    dot = np.dot(fp32_flat, int8_flat)
    norm_fp32 = np.linalg.norm(fp32_flat)
    norm_int8 = np.linalg.norm(int8_flat)
    cosine_sim = dot / (norm_fp32 * norm_int8 + 1e-8)
    
    return max_diff, mean_diff, cosine_sim


def main():
    print("=" * 50)
    print("第2周：量化精度评测")
    print("=" * 50)
    
    # 加载模型
    print("\n加载模型...")
    session_fp32 = ort.InferenceSession(MODEL_FP32, providers=['CPUExecutionProvider'])
    session_int8 = ort.InferenceSession(MODEL_INT8, providers=['CPUExecutionProvider'])
    
    # 取测试图片（用剩下的100张，不跟校准集重叠）
    all_files = sorted(os.listdir(TEST_DIR))
    test_files = all_files[200:300]  # 第201-300张做精度测试
    
    print(f"测试图片数: {len(test_files)}")
    
    max_diffs = []
    mean_diffs = []
    cosine_sims = []
    
    for i, fname in enumerate(test_files):
        img_path = os.path.join(TEST_DIR, fname)
        try:
            tensor, _ = preprocess_image(img_path)
        except Exception:
            continue
        
        fp32_out = run_inference(session_fp32, tensor)
        int8_out = run_inference(session_int8, tensor)
        
        max_d, mean_d, cos_sim = compare_outputs(fp32_out, int8_out)
        max_diffs.append(max_d)
        mean_diffs.append(mean_d)
        cosine_sims.append(cos_sim)
        
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(test_files)} 张...")
    
    # 汇总报告
    print("\n" + "=" * 50)
    print("精度评测报告")
    print("=" * 50)
    print(f"测试图片: {len(max_diffs)} 张")
    print(f"输出最大差异 - 均值: {np.mean(max_diffs):.6f}, 最大: {np.max(max_diffs):.6f}")
    print(f"输出平均差异 - 均值: {np.mean(mean_diffs):.6f}")
    print(f"余弦相似度   - 均值: {np.mean(cosine_sims):.6f}, 最小: {np.min(cosine_sims):.6f}")
    
    # 判定
    avg_cos = np.mean(cosine_sims)
    if avg_cos > 0.99:
        verdict = "✅ 精度损失极小，量化成功，可部署"
    elif avg_cos > 0.95:
        verdict = "⚠️ 精度有轻微损失，建议增加校准数据或使用QAT"
    else:
        verdict = "❌ 精度损失严重，需重新量化或检查模型"
    
    print(f"\n判定: {verdict}")
    
    # 保存报告
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(f"量化精度评测报告\n")
        f.write(f"={'='*40}\n")
        f.write(f"测试图片: {len(max_diffs)} 张\n")
        f.write(f"输出最大差异均值: {np.mean(max_diffs):.6f}\n")
        f.write(f"输出平均差异均值: {np.mean(mean_diffs):.6f}\n")
        f.write(f"余弦相似度均值: {np.mean(cosine_sims):.6f}\n")
        f.write(f"余弦相似度最小: {np.min(cosine_sims):.6f}\n")
        f.write(f"判定: {verdict}\n")
    
    print(f"\n报告已保存到 {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
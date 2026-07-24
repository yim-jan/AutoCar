"""第4周 任务2 v2：CLAHE增强对比度 + 模型对比测试"""
import sys
sys.path.insert(0, '/workspace/yolov5')

import torch
import cv2
import numpy as np
import os
from preprocess import preprocess_image
from utils.general import non_max_suppression, scale_boxes

MODEL_PATH = "/workspace/yolov5s.pt"
TEST_DIR = "/workspace/data/hard_samples"
OUTPUT_DIR = "/workspace/data/hard_samples_enhanced"

def enhance_clahe(img):
    """CLAHE 自适应直方图均衡化，增强局部对比度"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def main():
    print("=" * 50)
    print("第4周 v2：CLAHE 对比度增强评测")
    print("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n[1] 加载模型...")
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['model'].float()
    model.eval()
    
    hard_files = sorted(os.listdir(TEST_DIR))[:20]
    
    print(f"\n[2] 测试 {len(hard_files)} 张困难样本...")
    print(f"{'图片':<20} {'原始检出':<10} {'CLAHE检出':<10} {'提升':<10}")
    print("-" * 50)
    
    total_orig = 0
    total_clahe = 0
    
    for fname in hard_files:
        img_path = os.path.join(TEST_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 原始推理
        tensor, shape = preprocess_image(img_path)
        tensor = torch.from_numpy(tensor)
        with torch.no_grad():
            out = model(tensor)[0]
            det = non_max_suppression(out, conf_thres=0.25, iou_thres=0.45)
        orig_count = len(det[0]) if det[0] is not None else 0
        
        # CLAHE 增强后推理
        enhanced = enhance_clahe(img)
        enhanced_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(enhanced_path, enhanced)
        
        tensor_e, shape_e = preprocess_image(enhanced_path)
        tensor_e = torch.from_numpy(tensor_e)
        with torch.no_grad():
            out_e = model(tensor_e)[0]
            det_e = non_max_suppression(out_e, conf_thres=0.25, iou_thres=0.45)
        clahe_count = len(det_e[0]) if det_e[0] is not None else 0
        
        total_orig += orig_count
        total_clahe += clahe_count
        
        improvement = clahe_count - orig_count
        print(f"{fname:<20} {orig_count:<10} {clahe_count:<10} {improvement:+d}")
    
    print("-" * 50)
    print(f"合计: 原始检出 {total_orig} → CLAHE检出 {total_clahe}")
    diff = total_clahe - total_orig
    print(f"提升: {diff:+d} ({'↑ 有效' if diff > 0 else '→ 持平' if diff == 0 else '↓ 退化'})")
    
    if diff > 3:
        verdict = "✅ CLAHE 显著有效，建议作为低光照场景标配预处理"
    elif diff > 0:
        verdict = "⚠️ CLAHE 有轻微改善，建议结合更多增强方法叠加使用"
    else:
        verdict = "❌ 预处理无法解决，需从训练数据源头增加低光照标注样本"
    
    print(f"\n[3] 最终判定: {verdict}")
    print(f"增强样本保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

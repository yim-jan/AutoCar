"""第4周 任务2：低光照数据增强 + 模型对比测试"""
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

def adjust_brightness(img, factor):
    """调整图片亮度 factor>1变亮 factor<1变暗"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def main():
    print("=" * 50)
    print("第4周：低光照场景模型优化")
    print("=" * 50)
    
    # 加载模型
    print("\n[1] 加载模型...")
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['model'].float()
    model.eval()
    
    # 取困难样本测试
    hard_files = sorted(os.listdir(TEST_DIR))[:20]
    
    print(f"\n[2] 测试 {len(hard_files)} 张困难样本...")
    print(f"{'图片':<20} {'原始检出':<10} {'增强后检出':<10} {'提升':<10}")
    print("-" * 50)
    
    total_orig = 0
    total_enhanced = 0
    
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
        
        # 亮度增强后推理
        enhanced = adjust_brightness(img, 1.5)
        enhanced_path = f"/tmp/enhanced_{fname}"
        cv2.imwrite(enhanced_path, enhanced)
        
        tensor_e, shape_e = preprocess_image(enhanced_path)
        tensor_e = torch.from_numpy(tensor_e)
        with torch.no_grad():
            out_e = model(tensor_e)[0]
            det_e = non_max_suppression(out_e, conf_thres=0.25, iou_thres=0.45)
        enhanced_count = len(det_e[0]) if det_e[0] is not None else 0
        
        total_orig += orig_count
        total_enhanced += enhanced_count
        
        improvement = enhanced_count - orig_count
        print(f"{fname:<20} {orig_count:<10} {enhanced_count:<10} {improvement:+d}")
    
    print("-" * 50)
    print(f"合计: 原始检出 {total_orig} → 增强后检出 {total_enhanced}")
    print(f"提升: {total_enhanced - total_orig} ({'↑' if total_enhanced > total_orig else '持平'})")
    
    print("\n[3] 结论:")
    if total_enhanced > total_orig:
        print("✅ 亮度增强有效提升困难场景检出，建议加入训练Pipeline")
    else:
        print("⚠️ 简单亮度增强不够，需结合直方图均衡化或GAN增强")

if __name__ == "__main__":
    main()

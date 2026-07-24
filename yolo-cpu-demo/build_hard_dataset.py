"""第4周 任务1：挖掘低光照困难样本"""
import os
import cv2
import numpy as np
import shutil

TEST_DIR = "/workspace/data/coco_calib"
OUTPUT_DIR = "/workspace/data/hard_samples"

def compute_brightness(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_files = sorted(os.listdir(TEST_DIR))
    hard_files = []
    
    for fname in all_files:
        img_path = os.path.join(TEST_DIR, fname)
        brightness = compute_brightness(img_path)
        if brightness is not None and brightness < 80:
            hard_files.append((fname, brightness))
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, fname))
    
    print(f"总图片: {len(all_files)}")
    print(f"低光照样本(亮度<80): {len(hard_files)} 张")
    print(f"已复制到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
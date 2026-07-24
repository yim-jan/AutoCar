"""
第3周 任务2：特定场景评测 — 夜间+逆光
"""
import sys
sys.path.insert(0, '/workspace/yolov5')

import os
import numpy as np
import cv2
import torch
from preprocess import preprocess_image
from utils.general import non_max_suppression, scale_boxes
from infer import COCO_CLASSES, draw_boxes

MODEL_PATH = "/workspace/yolov5s.pt"
TEST_DIR = "/workspace/data/coco_calib"
OUTPUT_DIR = "/workspace/data/night_scene_results"
OUTPUT_REPORT = "/workspace/night_scene_report.txt"


def compute_brightness(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def is_backlight(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top_mean = np.mean(gray[:h//2, :])
    bottom_mean = np.mean(gray[h//2:, :])
    return (top_mean / (bottom_mean + 1e-5)) > 1.5


def main():
    print("=" * 50)
    print("第3周 任务2：夜间+逆光场景评测")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    print("\n加载模型...")
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['model'].float()
    model.eval()

    # 筛选图片
    all_files = sorted(os.listdir(TEST_DIR))[200:300]

    night_files = []
    backlight_files = []

    print("\n[1/3] 筛选夜间和逆光场景...")
    for fname in all_files:
        img_path = os.path.join(TEST_DIR, fname)
        brightness = compute_brightness(img_path)
        if brightness is None:
            continue
        if brightness < 60:
            night_files.append((fname, brightness))
        if is_backlight(img_path):
            backlight_files.append((fname, brightness))

    hard_scenes = list(set(night_files + backlight_files))
    print(f"  夜间: {len(night_files)} 张 | 逆光: {len(backlight_files)} 张 | 困难场景合计: {len(hard_scenes)} 张")

    # 推理
    print("\n[2/3] 评测困难场景...")
    total_detections = 0
    car_count = 0
    person_count = 0
    low_conf_count = 0
    saved = 0

    for fname, brightness in hard_scenes:
        img_path = os.path.join(TEST_DIR, fname)
        try:
            input_tensor, original_shape = preprocess_image(img_path)
        except Exception:
            continue

        input_tensor = torch.from_numpy(input_tensor)
        with torch.no_grad():
            output = model(input_tensor)[0]
            detections = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45)

        img = cv2.imread(img_path)
        if detections[0] is not None and len(detections[0]):
            detections[0][:, :4] = scale_boxes(input_tensor.shape[2:], detections[0][:, :4], img.shape).round()
            total_detections += len(detections[0])

            for det in detections[0]:
                cls_id = int(det[5])
                conf = float(det[4])
                if cls_id == 2:
                    car_count += 1
                elif cls_id == 0:
                    person_count += 1
                if conf < 0.5:
                    low_conf_count += 1

            if saved < 10:
                img = draw_boxes(img, detections[0])
                cv2.imwrite(os.path.join(OUTPUT_DIR, fname), img)
                saved += 1

    # 报告
    print("\n[3/3] 生成报告...")
    avg_det = total_detections / len(hard_scenes) if hard_scenes else 0
    low_rate = low_conf_count / total_detections * 100 if total_detections > 0 else 0

    report = f"""
{'='*50}
夜间+逆光场景专项评测报告
{'='*50}

场景统计:
  总测试图片: {len(all_files)} 张
  夜间场景: {len(night_files)} 张 (亮度<60)
  逆光场景: {len(backlight_files)} 张
  困难场景合计: {len(hard_scenes)} 张

检测结果:
  困难场景平均检出: {avg_det:.1f} 个/帧
  车辆检出: {car_count} | 行人检出: {person_count}
  低置信度(<0.5): {low_conf_count} ({low_rate:.1f}%)

可视化: {OUTPUT_DIR}
"""
    print(report)

    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)
    print(f"报告已保存: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
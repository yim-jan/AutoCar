import sys
sys.path.insert(0, '/workspace/yolov5')

import torch
import numpy as np
import cv2
import time
from preprocess import preprocess_image
from utils.general import non_max_suppression, scale_boxes

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def draw_boxes(img, detections):
    colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        color = colors[cls_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{COCO_CLASSES[cls_id]}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def main():
    image_path = "/workspace/test.jpg"
    output_path = "/workspace/result.jpg"
    model_path = "/workspace/yolov5s.pt"

    print("1. 加载 PyTorch 模型...")
    model = torch.load(model_path, map_location='cpu', weights_only=False)['model'].float()
    model.eval()
    print("   模型加载完成")

    print("2. 预处理...")
    input_tensor, original_shape = preprocess_image(image_path)
    input_tensor = torch.from_numpy(input_tensor)

    print("3. 推理...")
    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)[0]
        detections = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45)
        print(f"   耗时: {(time.time()-start)*1000:.1f}ms")

    img = cv2.imread(image_path)
    if detections[0] is not None and len(detections[0]):
        detections[0][:, :4] = scale_boxes(input_tensor.shape[2:], detections[0][:, :4], img.shape).round()
        img = draw_boxes(img, detections[0])
        print(f"   检测到 {len(detections[0])} 个物体:")
        for det in detections[0]:
            print(f"     {COCO_CLASSES[int(det[5])]}: {float(det[4]):.2f}")
    else:
        print("   未检测到物体")

    cv2.imwrite(output_path, img)
    print(f"4. 结果保存: {output_path}")

if __name__ == "__main__":
    main()

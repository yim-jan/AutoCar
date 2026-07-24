import cv2
import numpy as np

def preprocess_image(image_path):
    """
     读取图片，预处理成ONNX模型需要的格式
    输入: 图片路径
    输出: numpy数组, 形状(1, 3, 640, 640), dtype=float32, 值域[0,1]
    """

    # TODO 1: 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # TODO 2: 获取原始尺寸 (后面画框时需要还原坐标)
    original_h, original_w = img.shape[:2]

     # TODO 3: 将BGR转为RGB
    # 提示: cv2.imread读出来是BGR，但YOLO预训练时用的是RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # TODO 4: Resize到640x640
    # 提示: 用cv2.resize，注意我们不做letterbox（保持宽高比的黑边填充），直接拉伸
    img = cv2.resize(img, (640, 640))

    # TODO 5: 归一化到[0,1]
    # 提示: 像素值原本是0-255整数，转成float32后除以255.0
    img = img.astype(np.float32) / 255.0

     # TODO 6: 调整通道顺序 HWC -> NCHW
    # 提示: 用np.transpose，当前shape是(H,W,C)，目标shape是(C,H,W)，然后增加batch维度
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # CHW -> NCH

    # TODO 7: 确保内存连续
    img = np.ascontiguousarray(img, dtype=np.float32)
    
    return img, (original_h, original_w)

    # 测试：保存预处理后的张量，看看数值范围对不对
if __name__ == "__main__":
    # 先用一张纯白图片自测
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.imwrite("/workspace/test_white.jpg", test_img)
    
    tensor, _ = preprocess_image("/workspace/test_white.jpg")
    print(f"输出形状: {tensor.shape}")  # 应该是 (1, 3, 640, 640)
    print(f"数据类型: {tensor.dtype}")   # 应该是 float32
    print(f"数值范围: [{tensor.min():.3f}, {tensor.max():.3f}]")  # 应该是 [0.0, 1.0] 左右
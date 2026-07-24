import torch
import sys
sys.path.insert(0, 'yolov5')

# 导出 ONNX 模型
model = torch.load('yolov5s.pt', map_location='cpu', weights_only=False)['model'].float()
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy_input, 'yolov5s.onnx', opset_version=11,
                  input_names=['images'], output_names=['output'])
print('ONNX 导出完成: yolov5s.onnx')

"""
第3周 任务1：算子级性能剖析
分析 YOLOv5s FP32 vs INT8 各算子耗时，定位性能瓶颈
"""
import onnxruntime as ort
import numpy as np
import json
import time

MODEL_FP32 = "/workspace/yolov5s.onnx"
MODEL_INT8 = "/workspace/yolov5s_int8.onnx"
OUTPUT_REPORT = "/workspace/profile_report.txt"


def profile_model(model_path, label, num_runs=50):
    """剖析模型，返回算子耗时统计"""
    
    # 开启性能剖析
    options = ort.SessionOptions()
    options.enable_profiling = True
    
    session = ort.InferenceSession(
        model_path,
        options,
        providers=['CPUExecutionProvider']
    )
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    dummy = np.random.randn(*input_shape).astype(np.float32)
    
    # 预热
    for _ in range(5):
        session.run(None, {input_name: dummy})
    
    # 计时总推理
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy})
    total_time = time.time() - start
    
    # 结束 profiling，获取日志文件
    profile_file = session.end_profiling()
    
    avg_ms = (total_time / num_runs) * 1000
    
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"平均推理时间: {avg_ms:.2f} ms ({num_runs}次平均)")
    print(f"Profile 日志: {profile_file}")
    
    return profile_file, avg_ms


def parse_profile_log(profile_file):
    """解析 ONNX Runtime profile 日志，提取各算子耗时"""
    try:
        with open(profile_file, 'r') as f:
            content = f.read()
        
        # 查找 JSON 部分
        if '[' in content and ']' in content:
            start = content.find('[')
            end = content.rfind(']') + 1
            json_str = content[start:end]
            
            # 处理可能的格式问题
            json_str = json_str.replace('}\n{', '},{')
            
            records = json.loads(json_str)
            
            # 按算子类型聚合耗时
            op_stats = {}
            for rec in records:
                if rec.get('cat') == 'Node':
                    op_name = rec.get('name', 'unknown')
                    op_type = op_name.split('_')[0] if '_' in op_name else op_name
                    duration = rec.get('dur', 0)  # 微秒
                    
                    if op_type not in op_stats:
                        op_stats[op_type] = {'total_dur': 0, 'count': 0}
                    op_stats[op_type]['total_dur'] += duration
                    op_stats[op_type]['count'] += 1
            
            return op_stats
    except Exception as e:
        print(f"[警告] 解析 profile 日志失败: {e}")
    
    return None


def classify_bottleneck(op_type, duration_ms):
    """根据算子类型判断瓶颈类别"""
    compute_ops = ['Conv', 'Gemm', 'MatMul', 'Softmax', 'Sigmoid', 'Reshape']
    memory_ops = ['Transpose', 'Concat', 'Slice', 'Gather', 'Resize', 'Cast']
    
    if any(c in op_type for c in compute_ops):
        return "Compute Bound (计算密集)"
    elif any(m in op_type for m in memory_ops):
        return "Memory Bound (数据搬运)"
    else:
        return "待分析"


def main():
    print("=" * 50)
    print("第3周 任务1：算子级性能剖析")
    print("=" * 50)
    
    report_lines = []
    report_lines.append("YOLOv5s 算子级性能剖析报告")
    report_lines.append("=" * 50)
    
    for model_path, label in [(MODEL_FP32, "FP32"), (MODEL_INT8, "INT8")]:
        profile_file, avg_ms = profile_model(model_path, label)
        
        op_stats = parse_profile_log(profile_file)
        
        if op_stats:
            # 排序，取 Top 5
            sorted_ops = sorted(
                op_stats.items(),
                key=lambda x: x[1]['total_dur'],
                reverse=True
            )[:5]
            
            total_op_time = sum(v['total_dur'] for v in op_stats.values())
            
            print(f"\nTop 5 耗时算子:")
            print(f"{'算子':<20} {'耗时(ms)':<12} {'占比':<10} {'瓶颈类型'}")
            print("-" * 55)
            
            for op_type, stats in sorted_ops:
                dur_ms = stats['total_dur'] / 1000
                pct = stats['total_dur'] / total_op_time * 100 if total_op_time > 0 else 0
                bottleneck = classify_bottleneck(op_type, dur_ms)
                print(f"{op_type:<20} {dur_ms:<12.3f} {pct:<10.1f}% {bottleneck}")
        
        print()
    
    # 保存报告
    with open(OUTPUT_REPORT, 'w') as f:
        f.write('\n'.join(report_lines))
        f.write(f"\nFP32 平均推理时间: {avg_ms:.2f} ms\n")
    
    print(f"报告已保存到 {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
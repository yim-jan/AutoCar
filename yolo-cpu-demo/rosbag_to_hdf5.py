"""
任务2：数据格式转换小工具
读取KITTI raw data → 对齐时间戳 → 场景分类 → 写入HDF5 → 输出分布报告
"""
import os
import numpy as np
import h5py
import cv2
from datetime import datetime
from tqdm import tqdm

# 配置路径
RAW_DIR = "/workspace/data/raw/2011_09_26_drive_0001_sync"
OUTPUT_H5 = "/workspace/data/processed/2011_09_26_drive_0001.h5"
OUTPUT_REPORT = "/workspace/data/processed/distribution_report.txt"


def load_timestamps(sensor_dir):
    """读取传感器时间戳文件，返回时间戳列表（单位：秒）"""
    ts_file = os.path.join(sensor_dir, "timestamps.txt")
    timestamps = []
    with open(ts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                dt = datetime.strptime(line.split('.')[0], "%Y-%m-%d %H:%M:%S")
                sec = dt.hour * 3600 + dt.minute * 60 + dt.second + float("0." + line.split('.')[1])
                timestamps.append(sec)
    return timestamps


def load_gps_speed(oxts_dir, frame_id):
    """读取第frame_id帧的GPS前向速度，返回 km/h"""
    filename = os.path.join(oxts_dir, "data", f"{frame_id:010d}.txt")
    with open(filename, 'r') as f:
        fields = f.readline().strip().split()
    vf = float(fields[8])  # 前向速度 m/s
    return vf * 3.6


def classify_scene(speed_kmh):
    """根据速度分类场景"""
    if speed_kmh > 80:
        return "highway"
    elif speed_kmh >= 20:
        return "urban"
    else:
        return "parking"


def load_lidar_points(velodyne_dir, frame_id):
    """读取一帧激光雷达点云，返回 Nx4 numpy数组"""
    filename = os.path.join(velodyne_dir, "data", f"{frame_id:010d}.bin")
    if not os.path.exists(filename):
        return None
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    if points.shape[0] == 0:
        return None
    return points


def load_image(image_dir, frame_id):
    """读取一帧图像，返回 numpy数组 (H, W, 3)"""
    filename = os.path.join(image_dir, "data", f"{frame_id:010d}.png")
    if not os.path.exists(filename):
        return None
    img = cv2.imread(filename)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    # 准备路径
    image_dir = os.path.join(RAW_DIR, "image_02")
    velodyne_dir = os.path.join(RAW_DIR, "velodyne_points")
    oxts_dir = os.path.join(RAW_DIR, "oxts")

    # 用图像帧作为主时间线
    total_frames = len(os.listdir(os.path.join(image_dir, "data")))
    print(f"总帧数: {total_frames}")

    # 加载时间戳
    print("加载时间戳...")
    timestamps = load_timestamps(image_dir)

    # 统计变量
    scene_counts = {"highway": 0, "urban": 0, "parking": 0}
    speed_list = []
    lidar_point_counts = []
    drop_count = 0
    timestamp_backward_count = 0
    invalid_frame_count = 0
    prev_ts = -1

    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)

    # 打开HDF5文件
    print("开始转换...")
    with h5py.File(OUTPUT_H5, 'w') as h5f:
        for frame_id in tqdm(range(total_frames), desc="处理帧"):
            frame_name = f"{frame_id:010d}"

            # === 异常处理1: 时间戳回退 ===
            current_ts = timestamps[frame_id]
            if current_ts <= prev_ts:
                timestamp_backward_count += 1
                print(f"[警告] 帧{frame_name}: 时间戳回退 ({prev_ts:.4f} -> {current_ts:.4f})")
            prev_ts = current_ts

            # === 读取GPS速度 ===
            try:
                speed = load_gps_speed(oxts_dir, frame_id)
            except Exception:
                speed = -1.0  # 无效速度
                invalid_frame_count += 1

            # === 场景分类 ===
            if speed < 0:
                scene = "unknown"
            else:
                scene = classify_scene(speed)
                speed_list.append(speed)
                scene_counts[scene] += 1

            # === 读取图像 ===
            img = load_image(image_dir, frame_id)
            if img is None:
                drop_count += 1
                print(f"[警告] 帧{frame_name}: 图像缺失，跳过")
                continue

            # === 读取点云 ===
            lidar = load_lidar_points(velodyne_dir, frame_id)
            if lidar is not None:
                lidar_point_counts.append(lidar.shape[0])
            else:
                invalid_frame_count += 1
                lidar = np.array([])  # 空数组占位

            # === 写入HDF5 ===
            grp = h5f.create_group(frame_name)
            grp.create_dataset("camera_image", data=img, compression="gzip")
            if len(lidar) > 0:
                grp.create_dataset("lidar_points", data=lidar, compression="gzip")
            grp.attrs["gps_speed_kmh"] = speed
            grp.attrs["scene_type"] = scene
            grp.attrs["timestamp"] = current_ts

    # === 生成分布报告 ===
    print("生成分布报告...")
    total_valid = sum(scene_counts.values())
    with open(OUTPUT_REPORT, 'w') as rpt:
        rpt.write("=" * 50 + "\n")
        rpt.write("KITTI Raw Data 数据分布报告\n")
        rpt.write("=" * 50 + "\n\n")

        rpt.write(f"总帧数: {total_frames}\n")
        rpt.write(f"有效帧: {total_valid}\n")
        rpt.write(f"丢帧: {drop_count}\n")
        rpt.write(f"时间戳回退: {timestamp_backward_count}\n")
        rpt.write(f"无效帧(GPS/点云): {invalid_frame_count}\n\n")

        rpt.write("场景分布:\n")
        for scene in ["highway", "urban", "parking"]:
            cnt = scene_counts[scene]
            pct = cnt / total_valid * 100 if total_valid > 0 else 0
            rpt.write(f"  {scene}: {cnt}帧 ({pct:.1f}%)\n")

        if speed_list:
            rpt.write(f"\n速度统计 (km/h):\n")
            rpt.write(f"  均值: {np.mean(speed_list):.1f}\n")
            rpt.write(f"  方差: {np.var(speed_list):.1f}\n")
            rpt.write(f"  最大: {np.max(speed_list):.1f}\n")
            rpt.write(f"  最小: {np.min(speed_list):.1f}\n")

        if lidar_point_counts:
            rpt.write(f"\n点云点数统计:\n")
            rpt.write(f"  均值: {np.mean(lidar_point_counts):.1f}\n")
            rpt.write(f"  方差: {np.var(lidar_point_counts):.1f}\n")

    print(f"转换完成！HDF5: {OUTPUT_H5}")
    print(f"报告: {OUTPUT_REPORT}")

    # 打印摘要
    print("\n=== 摘要 ===")
    print(f"总帧: {total_frames}, 有效: {total_valid}, 丢帧: {drop_count}")
    for scene, cnt in scene_counts.items():
        print(f"  {scene}: {cnt}")


if __name__ == "__main__":
    main()
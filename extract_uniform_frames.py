from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# =========================
# 配置区
# =========================

# 每个视频抽取多少帧
NUM_FRAMES = 16

# 你的视频存放路径
video_root = Path("/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/videos")

# 抽帧后的图片保存路径
output_root = Path(f"/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/frames_{NUM_FRAMES}")


# =========================
# 主程序
# =========================

def extract_uniform_frames(video_path: Path, output_dir: Path, num_frames: int) -> bool:
    """
    从单个视频中均匀抽取 num_frames 帧，并保存为 jpg 图片。
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果已经抽过帧，则跳过，避免重复处理
    existing_frames = sorted(output_dir.glob("frame_*.jpg"))
    if len(existing_frames) >= num_frames:
        return True

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        print(f"无法读取视频帧数: {video_path}")
        cap.release()
        return False

    # 如果视频总帧数少于目标抽帧数，则最多抽 total_frames 帧
    actual_num_frames = min(num_frames, total_frames)

    # 在整个视频范围内均匀选择帧编号
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        actual_num_frames
    ).astype(int)

    saved_count = 0

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()

        if not ok:
            print(f"读取失败: {video_path}, frame={frame_idx}")
            continue

        save_path = output_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(save_path), frame)
        saved_count += 1

    cap.release()

    if saved_count == 0:
        print(f"没有成功保存任何帧: {video_path}")
        return False

    return True


def main():
    if not video_root.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_root}")

    video_files = []

    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]:
        video_files.extend(video_root.rglob(ext))

    video_files = sorted(video_files)

    print(f"视频目录: {video_root}")
    print(f"输出目录: {output_root}")
    print(f"抽帧数量: 每个视频 {NUM_FRAMES} 帧")
    print(f"找到视频数量: {len(video_files)}")

    if len(video_files) == 0:
        print("没有找到视频文件，请检查路径是否正确。")
        return

    success = 0
    failed = 0

    for video_path in tqdm(video_files, desc="Extracting frames"):
        video_name = video_path.stem
        output_dir = output_root / video_name

        ok = extract_uniform_frames(
            video_path=video_path,
            output_dir=output_dir,
            num_frames=NUM_FRAMES
        )

        if ok:
            success += 1
        else:
            failed += 1

    print("\n抽帧完成")
    print(f"成功视频数: {success}")
    print(f"失败视频数: {failed}")
    print(f"抽帧结果保存到: {output_root}")


if __name__ == "__main__":
    main()
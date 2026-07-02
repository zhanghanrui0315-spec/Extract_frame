import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# =========================
# 路径配置
# =========================

ROOT = Path("/home/zhang-hr/big_model/datasets/UrbanVideo-Bench")

MCQ_PATH = ROOT / "MCQ.parquet"

NUM_FRAMES = 16

FRAMES_ROOT = ROOT / f"frames_{NUM_FRAMES}"

OUT_PATH = ROOT / f"urbanvideo_frames_{NUM_FRAMES}_sft.jsonl"


def build_sample(video_id: str, question: str, answer: str, frame_paths: list[str]) -> dict:
    """
    构造一条 Qwen3-VL / Unsloth 可用的多模态 SFT 样本。
    同一个 content 列表中的多张 image，就是同一个视频的关键帧序列。
    """

    content = []

    # 按时间顺序加入视频关键帧
    for idx, frame_path in enumerate(frame_paths):
        content.append({
            "type": "image",
            "image": frame_path
        })

    # 加入文本问题
    content.append({
        "type": "text",
        "text": (
            f"以下 {len(frame_paths)} 张图片来自同一段视频，并按照时间先后顺序排列。"
            f"请将它们视为该视频的关键帧序列。\n"
            f"请根据这些关键帧回答下面的多选题，只输出正确选项字母，不要解释。\n\n"
            f"视频编号：{video_id}\n"
            f"{question}"
        )
    })

    sample = {
        "messages": [
            {
                "role": "user",
                "content": content
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": str(answer).strip()
                    }
                ]
            }
        ]
    }

    return sample


def main():
    if not MCQ_PATH.exists():
        raise FileNotFoundError(f"没有找到问答文件: {MCQ_PATH}")

    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(f"没有找到抽帧目录: {FRAMES_ROOT}")

    df = pd.read_parquet(MCQ_PATH)

    required_columns = {"video_id", "question", "answer"}
    if not required_columns.issubset(set(df.columns)):
        raise ValueError(f"MCQ.parquet 缺少必要字段，需要包含: {required_columns}")

    print(f"读取问答文件: {MCQ_PATH}")
    print(f"抽帧目录: {FRAMES_ROOT}")
    print(f"输出文件: {OUT_PATH}")
    print(f"原始问答样本数: {len(df)}")

    written = 0
    missing_frames = 0
    insufficient_frames = 0

    missing_video_ids = []

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building SFT JSONL"):
            video_id = str(row["video_id"]).strip()
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()

            # 例如 EmbodiedCity_1.mp4 -> EmbodiedCity_1
            video_stem = Path(video_id).stem

            frame_dir = FRAMES_ROOT / video_stem
            frame_paths = sorted(frame_dir.glob("frame_*.jpg"))

            if len(frame_paths) == 0:
                missing_frames += 1
                missing_video_ids.append(video_id)
                continue

            if len(frame_paths) < NUM_FRAMES:
                insufficient_frames += 1

            # 最多取 16 张；如果实际少于 16 张，就取已有的全部
            frame_paths = frame_paths[:NUM_FRAMES]
            frame_paths = [str(p.resolve()) for p in frame_paths]

            sample = build_sample(
                video_id=video_id,
                question=question,
                answer=answer,
                frame_paths=frame_paths
            )

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print("\n生成完成")
    print(f"成功写入样本数: {written}")
    print(f"缺少抽帧的样本数: {missing_frames}")
    print(f"帧数不足 {NUM_FRAMES} 的样本数: {insufficient_frames}")
    print(f"输出文件: {OUT_PATH}")

    if missing_video_ids:
        print("\n前 20 个缺少抽帧的 video_id:")
        for vid in sorted(set(missing_video_ids))[:20]:
            print(vid)


if __name__ == "__main__":
    main()

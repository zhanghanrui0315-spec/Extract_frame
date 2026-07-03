import random
from pathlib import Path


def split_jsonl_train_val(
    src_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    将一个 JSONL 数据集随机划分为训练集和验证集。

    参数：
    src_path: 原始 JSONL 文件路径
    val_ratio: 验证集比例，默认 0.1，即 10%
    seed: 随机种子，保证每次划分结果一致
    """

    src = Path(src_path)

    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")

    train_out = src.parent / "train.jsonl"
    val_out = src.parent / "val.jsonl"

    lines = src.read_text(encoding="utf-8").splitlines()

    if len(lines) == 0:
        raise ValueError(f"源文件为空: {src}")

    random.seed(seed)
    random.shuffle(lines)

    n_val = int(len(lines) * val_ratio)

    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    train_out.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_out.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print("划分完成")
    print(f"原始样本数: {len(lines)}")
    print(f"训练集样本数: {len(train_lines)}")
    print(f"验证集样本数: {len(val_lines)}")
    print(f"训练集保存到: {train_out}")
    print(f"验证集保存到: {val_out}")


def main():
    src_path = "/home/zhang-hr/big_model/datasets/urbanvideo_hf_upload/urbanvideo_frames_16_sft_relative.jsonl"

    split_jsonl_train_val(
        src_path=src_path,
        val_ratio=0.1,
        seed=42,
    )


if __name__ == "__main__":
    main()
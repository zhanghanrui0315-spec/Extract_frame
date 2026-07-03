import os

# 尽量缓解 PyTorch 显存碎片问题，需要在 torch 初始化 CUDA 前设置
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def extract_answer_letter(text: str):
    """
    从文本中提取 A/B/C/D/E/F/G 作为选择题答案。
    例如：
    "D" -> D
    "D." -> D
    "The answer is G." -> G
    """
    if text is None:
        return None

    text = text.strip().upper()

    # 优先匹配独立的 A/B/C/D/E/F/G
    match = re.search(r"\b[A-G]\b", text)
    if match:
        return match.group(0)

    # 兜底：如果模型输出类似 "G." 或 "答案G"
    match = re.search(r"[A-G]", text)
    if match:
        return match.group(0)

    return None


def get_first_model_device(model):
    """
    获取模型所在设备。
    对于 device_map='auto'，通常仍然可以取第一个参数所在设备。
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_images_in_message(user_message: dict, max_images: int | None):
    """
    限制每条样本中最多使用多少张图片。
    max_images=None 或 max_images<=0 表示不限制。
    """
    if max_images is None or max_images <= 0:
        return user_message

    new_content = []
    image_count = 0

    for item in user_message.get("content", []):
        if item.get("type") == "image":
            if image_count < max_images:
                new_content.append(item)
                image_count += 1
        else:
            new_content.append(item)

    return {
        "role": user_message["role"],
        "content": new_content,
    }


def count_images(user_message: dict):
    return sum(
        1
        for item in user_message.get("content", [])
        if isinstance(item, dict) and item.get("type") == "image"
    )


def load_model(model_dir: str, base_model: str):
    """
    加载基础模型 + Unsloth Studio 保存的 LoRA adapter。
    """
    model_dir = Path(model_dir)

    print("=" * 80)
    print("加载 processor:", base_model)
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
    )

    print("加载基础模型:", base_model)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    if (model_dir / "adapter_config.json").exists():
        print("检测到 LoRA adapter，正在加载微调权重:")
        print(model_dir)
        model = PeftModel.from_pretrained(base, str(model_dir))
    else:
        print("未检测到 adapter_config.json，尝试直接加载模型目录:")
        print(model_dir)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("模型加载完成")
    print("=" * 80)

    return model, processor


def safe_cleanup(local_vars: dict):
    """
    清理当前样本产生的 GPU 张量和 Python 对象引用。
    """
    for name in [
        "inputs",
        "generated_ids",
        "generated_ids_trimmed",
        "output_text",
    ]:
        if name in local_vars:
            try:
                del local_vars[name]
            except Exception:
                pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def evaluate(
    model,
    processor,
    val_path: str,
    max_samples: int | None = None,
    max_images: int | None = 16,
    max_new_tokens: int = 1,
    print_every: int = 20,
):
    val_path = Path(val_path)

    if not val_path.exists():
        raise FileNotFoundError(f"验证集文件不存在: {val_path}")

    model_device = get_first_model_device(model)

    total = 0
    correct = 0
    invalid_gold = 0
    invalid_pred = 0
    oom_count = 0

    print("=" * 80)
    print("开始评估")
    print("验证集:", val_path)
    print("最多评估样本数:", max_samples)
    print("每条样本最多使用图片数:", max_images)
    print("max_new_tokens:", max_new_tokens)
    print("模型设备:", model_device)
    print("=" * 80)

    with open(val_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if max_samples is not None and total >= max_samples:
                break

            try:
                sample = json.loads(line)

                raw_user_message = sample["messages"][0]
                user_message = limit_images_in_message(
                    raw_user_message,
                    max_images=max_images,
                )

                gold_text = sample["messages"][1]["content"][0]["text"]
                gold = extract_answer_letter(gold_text)

                if gold is None:
                    invalid_gold += 1
                    continue

                messages = [user_message]

                image_num = count_images(user_message)

                print(f"\n正在评估第 {total + 1} 条，JSONL 行号: {line_idx}，图片数: {image_num}")

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                if hasattr(inputs, "input_ids"):
                    print("input_ids shape:", tuple(inputs.input_ids.shape))

                # 将输入放到模型所在设备
                inputs = inputs.to(model_device)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                pred = extract_answer_letter(output_text)

                total += 1

                if pred is None:
                    invalid_pred += 1

                is_correct = pred == gold
                if is_correct:
                    correct += 1

                # 前 10 条打印详细结果
                if total <= 10:
                    print("-" * 80)
                    print(f"样本 {total}")
                    print("标准答案:", gold)
                    print("模型原始输出:", repr(output_text))
                    print("提取预测:", pred)
                    print("是否正确:", is_correct)

                if total % print_every == 0:
                    acc = correct / total if total > 0 else 0.0
                    print("\n" + "=" * 80)
                    print(f"已评估 {total} 条")
                    print(f"当前答对: {correct}")
                    print(f"当前准确率: {acc:.4f}，即 {acc * 100:.2f}%")
                    print(f"无效预测数: {invalid_pred}")
                    print(f"OOM 跳过数: {oom_count}")
                    print("=" * 80)

                safe_cleanup(locals())

            except torch.cuda.OutOfMemoryError as e:
                oom_count += 1
                print("\n" + "!" * 80)
                print(f"第 {line_idx} 行发生 CUDA OOM，已跳过该样本。")
                print("OOM 信息:", str(e).split("\n")[0])
                print("建议：如果 OOM 多次出现，请使用 --max_images 8 或更小。")
                print("!" * 80)

                safe_cleanup(locals())
                continue

            except Exception as e:
                print("\n" + "!" * 80)
                print(f"第 {line_idx} 行发生非 OOM 错误，程序停止。")
                print("错误类型:", type(e).__name__)
                print("错误信息:", repr(e))
                print("!" * 80)
                raise

    accuracy = correct / total if total > 0 else 0.0

    print("\n" + "=" * 80)
    print("评估完成")
    print("有效评估样本数:", total)
    print("答对数量:", correct)
    print("准确率:", f"{accuracy:.4f}")
    print("准确率百分比:", f"{accuracy * 100:.2f}%")
    print("标准答案无效数:", invalid_gold)
    print("模型输出无法提取 A/B/C/D/E 的数量:", invalid_pred)
    print("OOM 跳过样本数:", oom_count)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Unsloth Studio 输出的模型目录",
    )

    parser.add_argument(
        "--val_path",
        type=str,
        default="/home/zhang-hr/big_model/datasets/urbanvideo_frame_16/val_abs.jsonl",
        help="验证集 JSONL 文件路径",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        help="基础模型名称",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多评估多少条。测试时可以设置 5、20、50；完整评估时不设置。",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=16,
        help="每条样本最多使用多少张图片。默认 16；如果 OOM，可改成 8。",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1,
        help="模型最多生成多少个 token。选择题只需要 1。",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=20,
        help="每评估多少条打印一次当前准确率。",
    )

    args = parser.parse_args()

    model, processor = load_model(
        model_dir=args.model_dir,
        base_model=args.base_model,
    )

    evaluate(
        model=model,
        processor=processor,
        val_path=args.val_path,
        max_samples=args.max_samples,
        max_images=args.max_images,
        max_new_tokens=args.max_new_tokens,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()

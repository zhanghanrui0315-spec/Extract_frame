# UrbanVideo-Bench 视频问答数据预处理与 Qwen3-VL 评估流程

本项目用于将 UrbanVideo-Bench 原始视频问答数据整理为适合 **Qwen3-VL / Unsloth Studio** 进行多模态监督微调的数据格式，并提供训练后模型的验证集准确率评估脚本。

整体流程如下：

```text
原始视频 videos/
        ↓
extract_uniform_frames.py
        ↓
均匀抽帧 frames_16/
        ↓
build_urbanvideo_frames_16_sft.py
        ↓
多模态 SFT 数据 urbanvideo_frames_16_sft.jsonl
        ↓
split_train_val.py
        ↓
train.jsonl / val.jsonl
        ↓
Unsloth Studio 微调 Qwen3-VL
        ↓
eval_qwen3vl_accuracy.py
        ↓
验证集准确率评估
```

---

## 1. 文件说明

| 文件名                              | 作用                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| `extract_uniform_frames.py`         | 从原始视频中均匀抽取固定数量的关键帧，并保存为图片序列。     |
| `build_urbanvideo_frames_16_sft.py` | 将视频关键帧、问题文本和答案标签构造成 Qwen3-VL 可用的多模态 ChatML / SFT JSONL 数据。 |
| `split_train_val.py`                | 将生成的 JSONL 数据随机划分为训练集和验证集。                |
| `eval_qwen3vl_accuracy.py`          | 加载 Unsloth Studio 训练后的 LoRA 模型，并在验证集上计算选择题准确率。 |

---

## 2. 数据对齐关系说明

本数据集不是传统图像分类任务，而是**视频多模态问答任务**。

每条样本的对齐关系为：

```text
video_id 对应的视频抽帧文件夹
        +
question 中的问题文本和候选选项
        ↓
answer 中的正确选项字母
```

具体来说：

```text
frames_16/{video_id}/frame_000.jpg
frames_16/{video_id}/frame_001.jpg
...
frames_16/{video_id}/frame_015.jpg
        +
question
        ↓
answer，例如 A / B / C / D / E / F / G
```

因此，模型训练目标是：

```text
输入：同一视频的多帧关键帧 + 问题文本 + 候选选项
输出：正确选项字母
```

---

## 3. 环境依赖

建议在 Ubuntu 环境下运行。常用依赖如下：

```bash
pip install opencv-python numpy pandas pyarrow tqdm
```

如果需要运行训练后评估脚本，还需要安装：

```bash
pip install torch transformers peft
```

如果使用 Unsloth Studio 的虚拟环境，建议先激活：

```bash
source /home/zhang-hr/.unsloth/studio/unsloth_studio/bin/activate
```

---

## 4. 第一步：均匀抽取视频关键帧

脚本：

```bash
extract_uniform_frames.py
```

该脚本会从原始视频目录中读取视频文件，并为每个视频均匀抽取 `NUM_FRAMES` 张关键帧。

默认配置：

```python
NUM_FRAMES = 16

video_root = Path("/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/videos")

output_root = Path(f"/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/frames_{NUM_FRAMES}")
```

运行命令：

```bash
python extract_uniform_frames.py
```

运行后会生成类似目录：

```text
UrbanVideo-Bench/
├── videos/
│   ├── xxx.mp4
│   └── ...
└── frames_16/
    ├── xxx/
    │   ├── frame_000.jpg
    │   ├── frame_001.jpg
    │   ├── ...
    │   └── frame_015.jpg
    └── ...
```

脚本支持的视频格式包括：

```text
.mp4 / .avi / .mov / .mkv / .webm
```

如果某个视频已经完成抽帧，脚本会自动跳过，避免重复处理。

---

## 5. 第二步：构建 Qwen3-VL 多模态 SFT 数据

脚本：

```bash
build_urbanvideo_frames_16_sft.py
```

该脚本会读取：

```text
MCQ.parquet
frames_16/
```

然后根据 `video_id` 找到对应的视频帧文件夹，将多张图片、问题文本和答案标签构造成 Qwen3-VL / Unsloth 可用的多模态 SFT JSONL 文件。

默认配置：

```python
ROOT = Path("/home/zhang-hr/big_model/datasets/UrbanVideo-Bench")

MCQ_PATH = ROOT / "MCQ.parquet"

NUM_FRAMES = 16

FRAMES_ROOT = ROOT / f"frames_{NUM_FRAMES}"

OUT_PATH = ROOT / f"urbanvideo_frames_{NUM_FRAMES}_sft.jsonl"
```

运行命令：

```bash
python build_urbanvideo_frames_16_sft.py
```

生成文件：

```text
urbanvideo_frames_16_sft.jsonl
```

每一行是一条训练样本，格式类似：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/frame_000.jpg"},
        {"type": "image", "image": "/path/to/frame_001.jpg"},
        {"type": "text", "text": "以下 16 张图片来自同一段视频..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "D"}
      ]
    }
  ]
}
```

其中：

- `image`：视频关键帧路径；
- `text`：问题文本和提示词；
- `assistant` 中的 `text`：监督标签，即正确选项字母。

---

## 6. 第三步：划分训练集和验证集

脚本：

```bash
split_train_val.py
```

该脚本会将生成的 JSONL 文件随机划分为：

```text
train.jsonl
val.jsonl
```

默认划分比例：

```python
val_ratio = 0.1
```

即：

```text
90% 训练集
10% 验证集
```

默认路径：

```python
src_path = "/home/zhang-hr/big_model/datasets/urbanvideo_hf_upload/urbanvideo_frames_16_sft_relative.jsonl"
```

使用前需要根据自己的实际数据路径修改 `src_path`。例如：

```python
src_path = "/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/urbanvideo_frames_16_sft.jsonl"
```

运行命令：

```bash
python split_train_val.py
```

输出示例：

```text
划分完成
原始样本数: 5348
训练集样本数: 4814
验证集样本数: 534
训练集保存到: train.jsonl
验证集保存到: val.jsonl
```

---

## 7. 第四步：导入 Unsloth Studio 训练

在 Unsloth Studio 中选择本地数据集：

```text
train.jsonl
```

如果有验证集入口，可以同时选择：

```text
val.jsonl
```

推荐训练模型：

```text
unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit
```

推荐初始参数：

```text
Method: QLoRA
Batch Size: 1
Gradient Accumulation: 8
Learning Rate: 5e-5
LoRA Rank: 16
LoRA Alpha: 32
LoRA Dropout: 0.05
Max Steps: 1000 - 3000
Context Length: 2048 或 4096
Optimizer: Paged AdamW 8-bit
Scheduler: Cosine
```

如果显存压力较大，可以优先降低：

```text
Context Length
LoRA Rank
输入帧数
```

---

## 8. 第五步：评估训练后模型准确率

脚本：

```bash
eval_qwen3vl_accuracy.py
```

该脚本用于加载 Unsloth Studio 输出的模型目录，并在验证集上计算选择题准确率。

该评估脚本支持：

- 加载基础模型；
- 加载 LoRA adapter；
- 支持 A-G 选项答案提取；
- 每条样本限制最大图片数；
- `max_new_tokens=1`，只生成一个答案字母；
- `use_cache=False`，降低推理显存压力；
- 自动清理每条样本的 GPU 缓存；
- 遇到 CUDA OOM 时跳过该样本并继续评估。

运行前建议设置：

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

示例命令：

```bash
source /home/zhang-hr/.unsloth/studio/unsloth_studio/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_DIR=/home/zhang-hr/.unsloth/studio/outputs/unsloth_Qwen3-VL-4B-Instruct-unsloth-bnb-4bit_1783041665

python eval_qwen3vl_accuracy.py \
  --model_dir "$MODEL_DIR" \
  --val_path /home/zhang-hr/big_model/datasets/urbanvideo_frame_16/val_abs.jsonl \
  --max_images 16 \
  --max_new_tokens 1
```

如果只想快速测试前 20 条：

```bash
python eval_qwen3vl_accuracy.py \
  --model_dir "$MODEL_DIR" \
  --val_path /home/zhang-hr/big_model/datasets/urbanvideo_frame_16/val_abs.jsonl \
  --max_samples 20 \
  --max_images 16 \
  --max_new_tokens 1
```

如果出现显存不足，可以改为只使用前 8 张图评估：

```bash
python eval_qwen3vl_accuracy.py \
  --model_dir "$MODEL_DIR" \
  --val_path /home/zhang-hr/big_model/datasets/urbanvideo_frame_16/val_abs.jsonl \
  --max_images 8 \
  --max_new_tokens 1
```

---

## 9. 当前实验结果示例

当前模型在自行划分的验证集上得到：

```text
有效评估样本数: 534
答对数量: 185
准确率: 0.3464
准确率百分比: 34.64%
标准答案无效数: 0
模型输出无法提取 A/B/C/D/E 的数量: 0
OOM 跳过样本数: 0
```

该结果高于随机猜测和多数类基线，说明模型经过微调后已经学到一定的视频问答规律，但仍有较大提升空间。

---

## 10. 后续优化建议

### 10.1 重新生成 8 帧或 12 帧版本

当前 16 帧输入可能导致 token 数过长。建议尝试：

```python
NUM_FRAMES = 8
```

或：

```python
NUM_FRAMES = 12
```

重新执行：

```bash
python extract_uniform_frames.py
python build_urbanvideo_frames_16_sft.py
python split_train_val.py
```

注意：如果修改帧数，建议同步修改脚本文件名和输出目录，例如：

```text
frames_8/
urbanvideo_frames_8_sft.jsonl
```

---

### 10.2 将提示词改为英文

如果原始问题和选项为英文，建议将提示词也改为英文，例如：

```text
The following {N} images are uniformly sampled frames from the same video in chronological order.
Treat them as a video frame sequence.
Answer the multiple-choice question based on the frames.
Only output the correct option letter from the available choices, such as A, B, C, D, E, F, or G.
Do not explain.
```

---

### 10.3 按 video_id 分组划分数据集

如果同一个 `video_id` 对应多个问题，建议后续按照 `video_id` 分组划分训练集、验证集和测试集，避免同一视频同时出现在训练集和验证集中。

更规范的划分方式为：

```text
train: 80%
val: 10%
test: 10%
```

并保证：

```text
同一个 video_id 的所有问答样本只出现在一个数据子集中
```

---

### 10.4 建立原始模型 baseline

建议使用同一验证集评估原始 Qwen3-VL-4B-Instruct 模型，得到：

```text
原始模型准确率
微调后模型准确率
```

这样才能判断微调带来的真实提升幅度。

---

## 11. 注意事项

### 11.1 当前结果不是官方测试集结果

如果原始数据没有提供官方训练集、验证集、测试集划分，则当前结果应表述为：

```text
自行划分验证集上的阶段性准确率
```

不要表述为：

```text
测试集准确率
```

---

### 11.2 A-G 都是合法答案

当前数据中答案标签范围覆盖：

```text
A / B / C / D / E / F / G
```

因此评估脚本不能只支持 A-E，否则会把 F/G 样本误判为无效标准答案。

---

### 11.3 16 帧输入可能过长

评估时部分样本的输入长度可能超过 4000 甚至 10000 token。若训练时 Context Length 设置为 2048，可能会产生训练和评估输入长度不一致的问题。建议后续优先尝试 8 帧或 12 帧版本。

---

## 12. 推荐运行顺序汇总

```bash
# 1. 抽取视频关键帧
python extract_uniform_frames.py

# 2. 构造多模态 SFT JSONL
python build_urbanvideo_frames_16_sft.py

# 3. 划分训练集和验证集
python split_train_val.py

# 4. 在 Unsloth Studio 中导入 train.jsonl 进行 QLoRA 微调

# 5. 评估微调后模型
source /home/zhang-hr/.unsloth/studio/unsloth_studio/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_DIR=/home/zhang-hr/.unsloth/studio/outputs/your_model_output_dir

python eval_qwen3vl_accuracy.py \
  --model_dir "$MODEL_DIR" \
  --val_path /path/to/val_abs.jsonl \
  --max_images 16 \
  --max_new_tokens 1
```

---

## 13. 目录结构示例

推荐项目结构：

```text
UrbanVideo-Qwen3VL/
├── extract_uniform_frames.py
├── build_urbanvideo_frames_16_sft.py
├── split_train_val.py
├── eval_qwen3vl_accuracy.py
├── README.md
└── data/
    ├── videos/
    ├── frames_16/
    ├── MCQ.parquet
    ├── urbanvideo_frames_16_sft.jsonl
    ├── train.jsonl
    └── val.jsonl
```

大型视频、抽帧图片和 JSONL 数据不建议上传到 GitHub，建议上传到 Hugging Face Dataset 或云盘。

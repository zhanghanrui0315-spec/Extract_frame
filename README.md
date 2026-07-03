# Extract_frame
# UrbanVideo-Bench 抽帧与多模态 SFT 数据构建脚本

本项目包含两个 Python 脚本，用于将 UrbanVideo-Bench 视频数据处理成适合 Qwen3-VL / Unsloth 训练的多模态 SFT 数据。

## 文件说明

| 文件名                              | 功能                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| `extract_uniform_frames.py`         | 对视频进行均匀抽帧，将每个视频保存为若干张 JPG 图片。        |
| `build_urbanvideo_frames_16_sft.py` | 读取抽帧结果和 `MCQ.parquet` 问答文件，生成 ChatML 风格的多模态 SFT JSONL 文件。 |

整体处理流程如下：

```text
原始视频 videos/
        ↓
extract_uniform_frames.py
        ↓
抽帧图片 frames_16/
        ↓
build_urbanvideo_frames_16_sft.py
        ↓
urbanvideo_frames_16_sft.jsonl
```

---

## 环境依赖

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install opencv-python numpy tqdm pandas pyarrow
```

依赖说明：

| 依赖            | 作用                                   |
| --------------- | -------------------------------------- |
| `opencv-python` | 读取视频并保存图片。                   |
| `numpy`         | 生成均匀抽帧的帧编号。                 |
| `tqdm`          | 显示处理进度条。                       |
| `pandas`        | 读取 `MCQ.parquet` 问答文件。          |
| `pyarrow`       | 让 `pandas` 支持读取 `.parquet` 文件。 |

---

## 一、视频均匀抽帧：`extract_uniform_frames.py`

### 1. 脚本作用

该脚本会扫描视频目录中的视频文件，并对每个视频均匀抽取固定数量的帧。

默认情况下，每个视频抽取 16 帧，并保存到：

```text
/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/frames_16/
```

每个视频会生成一个同名文件夹，例如：

```text
frames_16/
├── EmbodiedCity_1/
│   ├── frame_000.jpg
│   ├── frame_001.jpg
│   ├── ...
│   └── frame_015.jpg
├── EmbodiedCity_2/
│   ├── frame_000.jpg
│   └── ...
```

### 2. 修改抽帧数量

在 `extract_uniform_frames.py` 中找到：

```python
NUM_FRAMES = 16
```

如果你想每个视频抽取 8 帧，改成：

```python
NUM_FRAMES = 8
```

如果你想每个视频抽取 24 帧，改成：

```python
NUM_FRAMES = 24
```

修改后，输出目录会自动变成：

```text
frames_8/
frames_24/
```

因为代码中输出路径写的是：

```python
output_root = Path(f"/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/frames_{NUM_FRAMES}")
```

### 3. 修改视频输入路径

在 `extract_uniform_frames.py` 中找到：

```python
video_root = Path("/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/videos")
```

这里表示原始视频所在目录。

例如，如果你的视频放在：

```text
/home/zhang-hr/data/my_videos
```

就改成：

```python
video_root = Path("/home/zhang-hr/data/my_videos")
```

注意：Linux 绝对路径前面要有 `/`。

### 4. 修改抽帧输出路径

在 `extract_uniform_frames.py` 中找到：

```python
output_root = Path(f"/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/frames_{NUM_FRAMES}")
```

这里表示抽帧图片保存目录。

例如，如果你想保存到：

```text
/home/zhang-hr/data/video_frames_16
```

可以改成：

```python
output_root = Path("/home/zhang-hr/data/video_frames_16")
```

或者仍然保留自动根据抽帧数量命名的方式：

```python
output_root = Path(f"/home/zhang-hr/data/frames_{NUM_FRAMES}")
```

### 5. 运行抽帧脚本

进入脚本所在目录：

```bash
cd /home/zhang-hr/PycharmProjects/extract_frames
```

运行：

```bash
python extract_uniform_frames.py
```

运行成功后，终端会输出类似：

```text
视频目录: /home/zhang-hr/big_model/datasets/UrbanVideo-Bench/videos
输出目录: /home/zhang-hr/big_model/datasets/UrbanVideo-Bench/frames_16
抽帧数量: 每个视频 16 帧
找到视频数量: 1392

抽帧完成
成功视频数: 1392
失败视频数: 0
抽帧结果保存到: /home/zhang-hr/big_model/datasets/UrbanVideo-Bench/frames_16
```

### 6. 支持的视频格式

脚本默认会搜索以下视频格式：

```python
["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
```

如果你的视频是其他格式，例如 `.flv`，可以在代码中添加：

```python
for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.flv"]:
    video_files.extend(video_root.rglob(ext))
```

### 7. 注意事项

如果某个视频已经抽过帧，并且输出目录中已有足够数量的 `frame_*.jpg`，脚本会自动跳过该视频，避免重复处理。

如果你修改了 `NUM_FRAMES`，建议使用新的输出目录，例如：

```text
frames_8/
frames_16/
frames_24/
```

不要把不同抽帧数量的结果混在同一个文件夹中。

---

## 二、生成多模态 SFT JSONL：`build_urbanvideo_frames_16_sft.py`

### 1. 脚本作用

该脚本读取：

```text
MCQ.parquet
frames_16/
```

然后生成多模态训练用的 JSONL 文件：

```text
urbanvideo_frames_16_sft.jsonl
```

每一行是一个训练样本，格式大致如下：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/frame_000.jpg"},
        {"type": "image", "image": "/path/to/frame_001.jpg"},
        {"type": "text", "text": "问题文本"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "B"}
      ]
    }
  ]
}
```

该格式可用于 Qwen3-VL / Unsloth 的多模态 SFT 训练。

### 2. 修改数据集根目录

在 `build_urbanvideo_frames_16_sft.py` 中找到：

```python
ROOT = Path("/home/zhang-hr/big_model/datasets/UrbanVideo-Bench")
```

这里表示 UrbanVideo-Bench 数据集根目录。

该目录下应包含：

```text
UrbanVideo-Bench/
├── MCQ.parquet
├── frames_16/
```

如果你的数据集放在其他位置，例如：

```text
/home/zhang-hr/data/UrbanVideo-Bench
```

就改成：

```python
ROOT = Path("/home/zhang-hr/data/UrbanVideo-Bench")
```

### 3. 修改问答文件路径

默认问答文件路径是：

```python
MCQ_PATH = ROOT / "MCQ.parquet"
```

也就是说，脚本默认会读取：

```text
/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/MCQ.parquet
```

如果你的问答文件名字不同，例如：

```text
train.parquet
```

可以改成：

```python
MCQ_PATH = ROOT / "train.parquet"
```

### 4. 修改抽帧数量

在 `build_urbanvideo_frames_16_sft.py` 中找到：

```python
NUM_FRAMES = 16
```

这里的数字必须和抽帧脚本中的 `NUM_FRAMES` 保持一致。

例如：

如果你在 `extract_uniform_frames.py` 中设置：

```python
NUM_FRAMES = 24
```

那么这里也要改成：

```python
NUM_FRAMES = 24
```

否则脚本会去找：

```text
frames_16/
```

而不是：

```text
frames_24/
```

### 5. 修改抽帧目录

默认抽帧目录是：

```python
FRAMES_ROOT = ROOT / f"frames_{NUM_FRAMES}"
```

也就是说，如果 `NUM_FRAMES = 16`，脚本会读取：

```text
UrbanVideo-Bench/frames_16/
```

如果你的抽帧图片不在这个目录，而是在：

```text
/home/zhang-hr/data/my_frames
```

可以直接改成：

```python
FRAMES_ROOT = Path("/home/zhang-hr/data/my_frames")
```

### 6. 修改输出 JSONL 路径

默认输出路径是：

```python
OUT_PATH = ROOT / f"urbanvideo_frames_{NUM_FRAMES}_sft.jsonl"
```

如果 `NUM_FRAMES = 16`，输出文件为：

```text
urbanvideo_frames_16_sft.jsonl
```

如果你想自定义输出路径，例如：

```text
/home/zhang-hr/data/urbanvideo_train.jsonl
```

可以改成：

```python
OUT_PATH = Path("/home/zhang-hr/data/urbanvideo_train.jsonl")
```

### 7. 运行 JSONL 生成脚本

进入脚本目录：

```bash
cd /home/zhang-hr/PycharmProjects/extract_frames
```

运行：

```bash
python build_urbanvideo_frames_16_sft.py
```

运行成功后，终端会输出类似：

```text
读取问答文件: /home/zhang-hr/big_model/datasets/UrbanVideo-Bench/MCQ.parquet
抽帧目录: /home/zhang-hr/big_model/datasets/UrbanVideo-Bench/frames_16
输出文件: /home/zhang-hr/big_model/datasets/UrbanVideo-Bench/urbanvideo_frames_16_sft.jsonl
原始问答样本数: 5348

生成完成
成功写入样本数: 5348
缺少抽帧的样本数: 0
帧数不足 16 的样本数: 0
```

---

## 三、如果要上传到 Hugging Face，建议改成相对路径

当前 `build_urbanvideo_frames_16_sft.py` 中生成图片路径时使用了：

```python
frame_paths = [str(p.resolve()) for p in frame_paths]
```

这会生成绝对路径，例如：

```text
/home/zhang-hr/big_model/datasets/UrbanVideo-Bench/frames_16/EmbodiedCity_1/frame_000.jpg
```

如果只在本机训练，这样没问题。

但如果要上传到 Hugging Face，建议改成相对路径，例如：

```text
frames_16/EmbodiedCity_1/frame_000.jpg
```

可以将这一行：

```python
frame_paths = [str(p.resolve()) for p in frame_paths]
```

改成：

```python
frame_paths = [str(p.relative_to(ROOT)) for p in frame_paths]
```

前提是 `FRAMES_ROOT` 在 `ROOT` 目录内部，例如：

```text
ROOT/frames_16/
```

修改后生成的 JSONL 更适合上传到 Hugging Face 数据集仓库。

---

## 四、常见修改示例

### 示例 1：每个视频抽取 8 帧

两个脚本都修改：

```python
NUM_FRAMES = 8
```

然后重新运行：

```bash
python extract_uniform_frames.py
python build_urbanvideo_frames_16_sft.py
```

最终会得到：

```text
frames_8/
urbanvideo_frames_8_sft.jsonl
```

### 示例 2：每个视频抽取 24 帧

两个脚本都修改：

```python
NUM_FRAMES = 24
```

然后重新运行：

```bash
python extract_uniform_frames.py
python build_urbanvideo_frames_16_sft.py
```

最终会得到：

```text
frames_24/
urbanvideo_frames_24_sft.jsonl
```

### 示例 3：更换数据集路径

如果你的数据集目录是：

```text
/home/zhang-hr/data/UrbanVideo-Bench
```

抽帧脚本中修改：

```python
video_root = Path("/home/zhang-hr/data/UrbanVideo-Bench/videos")
output_root = Path(f"/home/zhang-hr/data/UrbanVideo-Bench/frames_{NUM_FRAMES}")
```

JSONL 生成脚本中修改：

```python
ROOT = Path("/home/zhang-hr/data/UrbanVideo-Bench")
MCQ_PATH = ROOT / "MCQ.parquet"
FRAMES_ROOT = ROOT / f"frames_{NUM_FRAMES}"
OUT_PATH = ROOT / f"urbanvideo_frames_{NUM_FRAMES}_sft.jsonl"
```

---






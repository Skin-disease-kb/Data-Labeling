# 皮肤病图像标注工具

基于 Qwen3-VL-8B-Instruct 模型的皮肤病图像自动标注工具，能够自动检测并标注皮肤病灶的边界框。

## 快速开始

### 1. 安装依赖

#### 一键安装脚本（推荐）

```bash
python install.py
```

安装脚本会自动：
- 检测环境（Colab/本地）
- 安装 Unsloth 和所有依赖
- 安装指定版本的 transformers 和相关库

### 2. 验证安装

运行以下命令确认 PyTorch 能识别 GPU：

```bash
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"
```

应该输出：`CUDA 可用: True`

### 3. 准备数据

将待标注的皮肤病图像放在一个文件夹中，例如：

```
D:\Skin-disease\images\
  ├── lesion_001.jpg
  ├── lesion_002.png
  ├── lesion_003.jpg
  └── ...
```

支持的图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

### 4. 运行标注

```bash
python annotate_skin_disease.py --input ./images --output annotations.csv
```

## 使用说明

### 基本用法

```bash
python annotate_skin_disease.py --input <图像文件夹> --output <输出CSV>
```

### 常用参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入图像文件夹（必需） | - |
| `--output` | `-o` | 输出 CSV 文件路径 | `annotations.csv` |
| `--model` | - | 模型路径或 HuggingFace ID | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` |
| `--temperature` | - | 采样温度（越低越稳定） | `0.7` |
| `--visualize` | - | 生成可视化图像 | `True` |
| `--no-visualize` | - | 禁用可视化 | - |
| `--viz-dir` | - | 可视化输出目录 | `visualized` |

### 使用示例

```bash
# 基本使用
python annotate_skin_disease.py --input ./images --output annotations.csv

# 指定可视化文件夹
python annotate_skin_disease.py --input ./images --output annotations.csv --viz-dir ./results

# 调整温度参数（更稳定的输出）
python annotate_skin_disease.py --input ./images --temperature 0.5

# 只标注不生成可视化
python annotate_skin_disease.py --input ./images --no-visualize

# 查看帮助
python annotate_skin_disease.py --help
```

## 输出说明

### 1. CSV 标注文件

生成的 CSV 文件包含以下字段：

| 字段 | 说明 |
|------|------|
| `image_name` | 图像文件名 |
| `image_path` | 图像完整路径 |
| `x1` | 左上角 X 坐标（相对 0-1） |
| `y1` | 左上角 Y 坐标（相对 0-1） |
| `x2` | 右下角 X 坐标（相对 0-1） |
| `y2` | 右下角 Y 坐标（相对 0-1） |
| `status` | 标注状态（success/failed） |
| `error_msg` | 错误信息（如果失败） |

**示例输出**：

```csv
image_name,image_path,x1,y1,x2,y2,status,error_msg
lesion_001.jpg,D:/Skin-disease/images/lesion_001.jpg,0.2345,0.3456,0.6789,0.8901,success,
lesion_002.png,D:/Skin-disease/images/lesion_002.png,0.1234,0.2345,0.5678,0.7890,success,
lesion_003.jpg,D:/Skin-disease/images/lesion_003.jpg,,,failed,无法从回复中解析边界框
```

### 2. 可视化图像

在 `visualized/` 文件夹中生成带红色边界框的图像，文件名与原图保持一致。

### 3. 错误日志

如果有失败的标注，会自动生成 `annotation_errors.log` 文件，记录所有失败的案例和错误信息。

## 坐标系统

- 使用**相对坐标**（0-1 范围）
- 原点在**左上角**
- 格式：`[x1, y1, x2, y2]`
  - `x1, y1`: 左上角坐标
  - `x2, y2`: 右下角坐标

转换为像素坐标：
```python
像素坐标 = 相对坐标 × 图像尺寸
abs_x1 = int(x1 × 图像宽度)
abs_y1 = int(y1 × 图像高度)
```

### 提示词策略

```
Please detect and localize the skin disease lesion in this image.

Requirements:
1. Output only the bounding box coordinates in JSON format
2. Use relative coordinates (0-1 range)
3. Format: {"bbox": [x1, y1, x2, y2]}
4. Return only the JSON, no additional text

Example output:
{"bbox": [0.2, 0.3, 0.8, 0.9]}
```

## 文件说明

```
D:\Skin-disease\
├── annotate_skin_disease.py   # 主标注脚本
├── install.py                 # 一键安装脚本
├── README.md                  # 本文档
├── images/                    # 输入图像文件夹（需自行创建）
├── annotations.csv            # 输出标注文件（运行后生成）
├── annotation_errors.log      # 错误日志（运行后生成）
└── visualized/                # 可视化图像（运行后生成）
```


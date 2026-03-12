# qwen-derm

基于 Qwen3-VL / Qwen2.5-VL 的皮肤病项目，当前包含两条主要工作流：

- 病灶定位与标注：`annotate_skin_disease.py`
- 原图分类 LoRA 微调：`train_qwen3_vl_lora_cls.py`

当前训练目标是：在尽量保留模型病灶定位能力的前提下，用 `data/` 原图数据集提升皮肤病分类能力。

## 项目结构

```text
qwen-derm/
├── annotate_skin_disease.py       # 病灶定位 / 标注脚本
├── train_qwen3_vl_lora_cls.py     # 原图分类 LoRA 微调脚本
├── requirements.txt               # 推理依赖（仅标注）
├── requirements-full.txt          # 完整依赖（标注 + 训练）
├── data/                          # 原图分类训练数据
├── visualized_images/             # 高亮病灶图数据（当前训练脚本未使用）
├── outputs/                       # 训练输出目录（运行后生成）
├── visualized/                    # 标注可视化目录（运行后生成）
└── .cache/                        # Triton / TorchInductor 缓存（运行后生成）
```

## 环境配置

### 推荐环境

- Python 3.10
- Linux 服务器 + NVIDIA GPU + CUDA
- 独立 conda 环境或 venv

### 训练环境安装

`requirements-full.txt` 是训练脚本使用的依赖文件。安装顺序固定为两步：

1. 先安装与服务器 CUDA 匹配的 PyTorch
2. 再安装项目依赖

```bash
conda create -n qwen-derm-lora python=3.10 -y
conda activate qwen-derm-lora

python -m pip install --upgrade pip setuptools wheel

# 示例：CUDA 12.6
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

# 安装训练依赖
python -m pip install -r requirements-full.txt
```

说明：

- `requirements-full.txt` 不直接写死 `torch / torchvision / torchaudio / triton / xformers / bitsandbytes`
- 这些底层包与服务器的 CUDA、驱动、系统和 GPU 架构强相关，必须按目标机器单独安装
- 如果服务器 CUDA 不是 `12.6`，把上面的 PyTorch 安装命令换成对应版本即可

### 仅标注环境安装

如果只做病灶定位与可视化，不跑训练：

```bash
conda create -n qwen-derm python=3.10 -y
conda activate qwen-derm

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
python -m pip install -r requirements.txt
```

### 安装验证

```bash
python -c "import torch, transformers; print('CUDA:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('transformers:', transformers.__version__)"
python -c "import unsloth; from trl import SFTTrainer, SFTConfig; print('unsloth+trl ok')"
```

至少应满足：

- `torch.cuda.is_available() == True`
- `transformers == 4.57.6`
- `unsloth` 和 `trl` 能正常导入

## 数据准备

### 训练数据

当前分类 LoRA 脚本使用 `data/` 原图数据集，目录名映射为 5 个固定标签：

```text
data/
├── ACK_光化性角化病/
├── DF_皮肤纤维瘤/
├── MEL_黑色素瘤/
├── SCC_鳞状细胞癌/
└── SEK_脂溢性角化病/
```

说明：

- 训练标签固定为：`ACK / DF / MEL / SCC / SEK`
- 空目录 `SEK_脂溢性角化病（高精图片版）` 会被自动忽略
- 训练脚本当前不使用 `visualized_images/` 里的高亮图

### 标注输入数据

标注脚本接收一个普通图像目录，例如：

```text
images/
├── lesion_001.jpg
├── lesion_002.png
└── ...
```

支持格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

## 分类 LoRA 微调

### 基本训练命令

```bash
python train_qwen3_vl_lora_cls.py
```

常用自定义路径：

```bash
python train_qwen3_vl_lora_cls.py \
  --data-dir ./data \
  --output-dir ./outputs/qwen3_vl_skin_cls_lora_raw
```

### 当前默认训练配置

当前脚本是一个偏保守的 LoRA 基线版本，默认配置如下：

- 基座模型：`unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`
- 加载方式：`load_in_4bit=True`
- 梯度检查点：`use_gradient_checkpointing="unsloth"`
- LoRA 范围：只训练语言层
- 视觉层：冻结
- 语言 attention：开启 LoRA
- 语言 MLP：开启 LoRA
- `r=16`
- `lora_alpha=16`
- `lora_dropout=0`
- `epochs=3`
- `learning_rate=1e-4`
- `max_length=1024`
- `batch_size=1`
- `gradient_accumulation_steps=8`
- 有效 batch size：`8`
- `warmup_ratio=0.05`
- `optim="adamw_8bit"`
- `weight_decay=0.01`
- `max_grad_norm=0.3`
- `lr_scheduler_type="cosine"`
- 自动使用 `bf16`；若 GPU 不支持则退到 `fp16`

数据处理默认配置：

- 验证集比例：`0.1`
- 随机种子：`3407`
- 少数类过采样下限：训练集中最大类样本数的 `0.5`

训练前后会自动生成验证集评估，并在每个 epoch 结束后保存当前最优 adapter。

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-dir` | 原始图像数据目录 | `D:\qwen-derm\data` |
| `--output-dir` | 训练输出目录 | `D:\qwen-derm\outputs\qwen3_vl_skin_cls_lora_raw` |
| `--model-name` | 基础模型 ID 或路径 | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` |
| `--fallback-model-name` | 8B 显存不足时的回退模型 | 空 |
| `--val-ratio` | 验证集比例 | `0.1` |
| `--epochs` | 训练轮数 | `3` |
| `--lr` | 学习率 | `1e-4` |
| `--max-length` | 最大序列长度 | `1024` |
| `--batch-size` | 单卡 batch size | `1` |
| `--grad-accum` | 梯度累积步数 | `8` |
| `--oversample-floor-ratio` | 少数类过采样下限比例 | `0.5` |
| `--max-steps` | 仅用于 smoke test | 空 |

### 推荐训练命令

默认保守版：

```bash
python train_qwen3_vl_lora_cls.py
```

服务器显存更充裕时，优先先提吞吐，不先改学习动态：

```bash
python train_qwen3_vl_lora_cls.py --batch-size 2 --grad-accum 4
```

如果仍然稳定，再试：

```bash
python train_qwen3_vl_lora_cls.py --batch-size 4 --grad-accum 2
```

说明：

- 上面两组都保持有效 batch size 为 `8`
- 这意味着训练稳定性与默认配置接近，但吞吐更高，更适合显存充足的服务器
- 当前脚本默认不会自动回退到 3B，只有显式传入 `--fallback-model-name` 才会启用回退

### 参数调节建议

#### 1. 显存优先策略

如果你的目标是“先稳妥跑通”：

- 保持 `4bit + language-only LoRA`
- 保持 `batch_size * grad_accum = 8`
- 先跑 `3` 个 epoch 看验证集 `macro_f1`

如果你的目标是“在服务器上更快”：

- 优先加 `batch-size`
- 对应减小 `grad-accum`
- 第一轮不要同时改学习率

推荐起点：

| 显存情况 | 建议参数 |
|----------|----------|
| 8GB 左右 | `--batch-size 1 --grad-accum 8` |
| 16GB - 24GB | `--batch-size 2 --grad-accum 4` |
| 24GB - 48GB | `--batch-size 4 --grad-accum 2` |

#### 2. 学习率策略

当前默认 `1e-4` 是稳妥起点。

- 如果只是提 `batch-size`，先不动学习率
- 如果后续把有效 batch 从 `8` 提到 `16`，仍然建议先继续用 `1e-4`
- 只有当训练很稳且收敛偏慢时，再考虑试 `1.5e-4`

#### 3. 训练轮数策略

- 第一轮正式训练建议保持 `3` 个 epoch
- 如果验证集 `macro_f1` 还在持续上升，再试 `5` 个 epoch
- 如果第 `2` 到 `3` 个 epoch 已经明显平台期，就不要盲目拉长

#### 4. 烟雾测试

不要一开始就直接跑完整训练。先跑一轮 smoke test：

```bash
python train_qwen3_vl_lora_cls.py --max-steps 2
```

确认这些都正常后，再跑正式训练：

- 模型能装入显存
- baseline 评估能正常结束
- 训练过程不 OOM
- `best_adapter` 和 `final_adapter` 能正确写出

#### 5. 大显存服务器的后续优化方向

当前脚本的重点是“先建立稳妥基线”，不是“榨干大显存”。

如果后续你想进一步利用服务器显存换取更强分类能力，优先级建议是：

1. 先把吞吐调上去：`batch-size ↑, grad-accum ↓`
2. 跑完完整训练，确认基线 `macro_f1`
3. 再考虑单独开实验，放开部分 vision layers 的 LoRA

注意：

- 第 3 步不是当前脚本默认行为
- 它需要额外改脚本，不能直接靠现有 CLI 参数打开

### 训练输出说明

训练输出目录默认在 `outputs/qwen3_vl_skin_cls_lora_raw/`，主要会生成：

- `label_mapping.json`
- `split_summary.json`
- `trainable_params.json`
- `model_resolution.json`
- `baseline_summary.json`
- `training_summary.json`
- `eval/baseline/`
- `eval/final/`
- `best_adapter/`
- `final_adapter/`

关键指标：

- `accuracy`
- `macro_f1`
- `invalid_rate`
- `classification_report`
- `confusion_matrix`

## 病灶标注脚本

### 基本用法

```bash
python annotate_skin_disease.py --input <图像文件夹> --output <输出CSV>
```

### 常用参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入图像文件夹 | 必填 |
| `--output` | `-o` | 输出 CSV 文件路径 | `annotations.csv` |
| `--model` | - | 模型路径或 HuggingFace ID | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` |
| `--temperature` | - | 采样温度 | `0.7` |
| `--visualize` | - | 生成可视化图像 | `True` |
| `--no-visualize` | - | 禁用可视化 | - |
| `--viz-dir` | - | 可视化输出目录 | `visualized` |

说明：

- 如果默认 `8B` 模型在显存不足时加载失败，脚本会自动回退到 `unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit`
- 脚本会把 Triton / TorchInductor 缓存写入项目目录下的 `.cache/`

### 使用示例

```bash
python annotate_skin_disease.py --input ./images --output annotations.csv
python annotate_skin_disease.py --input ./images --output annotations.csv --viz-dir ./results
python annotate_skin_disease.py --input ./images --temperature 0.5
python annotate_skin_disease.py --input ./images --model unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit
python annotate_skin_disease.py --input ./images --no-visualize
python annotate_skin_disease.py --help
```

### 标注输出

CSV 输出字段：

- `image_name`
- `image_path`
- `x1`
- `y1`
- `x2`
- `y2`
- `status`
- `error_msg`

可视化结果会输出到 `visualized/`，失败样本会记录到 `annotation_errors.log`。

## 版本说明

当前训练依赖采用以下已验证组合：

- `unsloth>=2026.2.1`
- `transformers==4.57.6`
- `trl==0.22.2`

其中：

- `transformers==4.57.6` 是为了稳定支持 `qwen3_vl`
- `trl==0.22.2` 与当前 `transformers` 组合已验证可用
- `requirements-full.txt` 会继续优先保证训练脚本可运行，而不是把所有 CUDA 底层轮子硬编码进去

## 许可证

MIT License

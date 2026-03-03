# qwen-derm 依赖冲突解决总报告

**报告日期**: 2026-03-03  
**适用范围**: `annotate_skin_disease.py` 推理标注流程（Windows + NVIDIA GPU）

---

## 1. 结论摘要

本项目此前两份冲突分析文档中的核心结论已部分过时。  
当前已完成依赖策略与脚本兼容修复，项目可在本机环境稳定运行，建议以本报告作为唯一依赖冲突说明文档。

---

## 2. 历史冲突与当前状态

| 历史冲突点 | 旧结论 | 当前状态 | 处理结果 |
|---|---|---|---|
| `unsloth` 与 `transformers` 版本冲突 | 曾建议锁到 `transformers<=4.56.2` | 当前实测 `unsloth 2026.2.1 + transformers 4.57.6` 可用 | ✅ 已解决 |
| `Qwen3-VL` 架构无法识别 | `transformers 4.56.2` 不识别 `qwen3_vl` | 已升级为 `transformers==4.57.6` | ✅ 已解决 |
| 8B 模型在 8GB 显存下加载失败 | 可能直接报错退出 | 脚本增加自动回退到 `Qwen2.5-VL-3B` | ✅ 已解决 |
| Windows 中文用户名导致 Triton/Inductor 缓存报编码错 | 运行期 `UnicodeDecodeError` | 脚本固定缓存到项目内 `.cache/`（ASCII 路径） | ✅ 已解决 |
| `--no-deps` 造成依赖缺失 | 旧流程有此风险 | 当前 `requirements*.txt` 为正常依赖安装策略 | ✅ 已规避 |

---

## 3. 当前统一依赖基线

### 推理环境（`requirements.txt`）

- `unsloth>=2026.2.1`
- `transformers==4.57.6`
- `Pillow>=9.0.0`
- `huggingface-hub>=0.23.0`

### 完整环境（`requirements-full.txt`）

- 推理环境全部依赖 +
- `trl==0.22.2`
- `datasets>=2.14.0`
- `tyro>=0.5.7`
- `rich>=13.0.0`

---

## 4. 推荐安装顺序（Windows）

1. 创建并激活 Conda 环境：
   - `conda create -n qwen-derm python=3.10 -y`
   - `conda activate qwen-derm`
2. 先安装 CUDA 版 PyTorch（示例 CU126）：
   - `python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio`
3. 再安装项目依赖：
   - 推理：`python -m pip install -r requirements.txt`
   - 完整：`python -m pip install -r requirements-full.txt`

---

## 5. 本次验证结果（实测）

### 环境版本检查

- `torch 2.10.0+cu126`
- `cuda_available True`
- `cuda_version 12.6`
- `transformers 4.57.6`
- `unsloth 2026.2.1`

### 脚本运行检查

- 命令：`python annotate_skin_disease.py --input .\tmp_test_images --output .\tmp_annotations.csv --no-visualize`
- 结果：默认 `8B` 模型显存不足时自动回退 `3B`，流程成功完成。
- 输出：`tmp_annotations.csv` 生成且 `status=success`。

---

## 6. 仍需关注的运行约束

1. 8GB 显存设备不保证稳定承载 8B 模型，建议保留自动回退逻辑。  
2. 首次运行会下载模型并编译缓存，耗时明显高于后续运行。  
3. 若切换 CUDA 版本或显卡驱动，建议重新验证 `torch` 与 `cuda_available`。

---

## 7. 文档收敛说明

以下历史文档已被本报告替代：

- `dependency_conflict_analysis.md`
- `DEPENDENCY_CONFLICTS_REPORT.md`

后续请仅维护本文件，避免多份冲突报告结论不一致。

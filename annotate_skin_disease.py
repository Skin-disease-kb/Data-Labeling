# -*- coding: utf-8 -*-
"""
皮肤病图像标注脚本

本脚本使用 Qwen3-VL-8B-Instruct 模型自动检测并定位皮肤病灶，
生成边界框标注。

输出格式: CSV 文件，包含相对坐标（0-1 范围）
可视化: 在原图上绘制红色边界框
"""

import argparse
import json
import re
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List, Dict

from PIL import Image, ImageDraw
from unsloth import FastVisionModel
import torch


# ============================================================================
# 模块 A: 模型加载函数
# ============================================================================

def load_model(model_path: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit") -> tuple:
    """
    加载 Qwen3-VL 模型用于推理。

    基于 qwen3_vl_(8b)_vision.py:75-79

    参数:
        model_path: HuggingFace 模型标识符或本地路径

    返回:
        (model, tokenizer) 元组
    """
    print(f"正在加载模型: {model_path}")
    print("使用 4-bit 量化以降低内存占用...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # 启用推理模式
    FastVisionModel.for_inference(model)

    print("模型加载成功！")
    return model, tokenizer


# ============================================================================
# 模块 B: 图像标注函数
# ============================================================================

def annotate_single_image(
    image: Image.Image,
    model,
    tokenizer,
    temperature: float = 0.7,
    max_new_tokens: int = 256
) -> str:
    """
    对单张图像进行边界框预测标注。

    基于 qwen3_vl_(8b)_vision.py:148-170

    参数:
        image: PIL 图像对象（RGB 模式）
        model: 已加载的 Qwen3-VL 模型
        tokenizer: 已加载的分词器
        temperature: 采样温度（越低越确定性）
        max_new_tokens: 最大生成 token 数

    返回:
        包含边界框预测的助手回复文本
    """
    # 构造指令提示词
    instruction = """Please detect and localize the skin disease lesion in this image.

Requirements:
1. Output only the bounding box coordinates in JSON format
2. Use relative coordinates (0-1 range)
3. Format: {"bbox": [x1, y1, x2, y2]}
- x1, y1: top-left corner
- x2, y2: bottom-right corner
4. Return only the JSON, no additional text

Example output:
{"bbox": [0.2, 0.3, 0.8, 0.9]}"""

    # 构造消息格式
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]

    # 应用聊天模板
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    # 分词处理输入
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=0.1,
            use_cache=True
        )

    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取助手的回复（在指令之后）
    # 找到助手回复开始的位置
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]

    return response.strip()


# ============================================================================
# 模块 C: 边界框解析器
# ============================================================================

def parse_bbox_from_text(text: str) -> Optional[List[float]]:
    """
    从模型输出中解析边界框坐标。

    支持多种格式:
    - JSON: {"bbox": [x1, y1, x2, y2]}
    - 数组: [x1, y1, x2, y2]
    - 文本描述

    参数:
        text: 模型输出文本

    返回:
        [x1, y1, x2, y2] 列表，解析失败则返回 None
    """
    # 尝试提取 JSON 格式
    json_pattern = r'\{\s*"bbox"\s*:\s*\[([^\]]+)\]\s*\}'
    json_match = re.search(json_pattern, text)

    if json_match:
        coords_str = json_match.group(1)
        coords = [float(x.strip()) for x in coords_str.split(',')]
        if len(coords) == 4:
            return coords

    # 尝试直接 JSON 解析
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "bbox" in data:
            return list(data["bbox"])
        elif isinstance(data, list) and len(data) == 4:
            return list(data)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 尝试查找数组模式 [x1, y1, x2, y2]
    array_pattern = r'\[([0-9.,\s]+)\]'
    array_match = re.search(array_pattern, text)

    if array_match:
        coords_str = array_match.group(1)
        coords = [float(x.strip()) for x in coords_str.split(',')]
        if len(coords) == 4:
            return coords

    return None


def validate_bbox(bbox: List[float]) -> Tuple[bool, Optional[str]]:
    """
    验证边界框坐标的合法性。

    参数:
        bbox: [x1, y1, x2, y2] 列表

    返回:
        (是否合法, 错误信息) 元组
    """
    if len(bbox) != 4:
        return False, f"边界框长度无效: {len(bbox)}，应为 4"

    x1, y1, x2, y2 = bbox

    # 检查 NaN 或 Inf
    if any(not isinstance(c, (int, float)) or c != c for c in bbox):
        return False, "边界框包含 NaN 或 Inf"

    # 允许轻微超出范围（0-1 范围，±0.1 容差）
    tolerance = 0.1
    if not (-tolerance <= x1 <= 1 + tolerance):
        return False, f"x1={x1} 超出有效范围"
    if not (-tolerance <= y1 <= 1 + tolerance):
        return False, f"y1={y1} 超出有效范围"
    if not (-tolerance <= x2 <= 1 + tolerance):
        return False, f"x2={x2} 超出有效范围"
    if not (-tolerance <= y2 <= 1 + tolerance):
        return False, f"y2={y2} 超出有效范围"

    # 将坐标限制到 [0, 1] 范围
    bbox[0] = max(0.0, min(1.0, x1))
    bbox[1] = max(0.0, min(1.0, y1))
    bbox[2] = max(0.0, min(1.0, x2))
    bbox[3] = max(0.0, min(1.0, y2))

    # 检查 x2 > x1 且 y2 > y1
    if bbox[2] <= bbox[0]:
        return False, f"x2={bbox[2]} 必须大于 x1={bbox[0]}"
    if bbox[3] <= bbox[1]:
        return False, f"y2={bbox[3]} 必须大于 y1={bbox[1]}"

    return True, None


# ============================================================================
# 模块 D: 可视化函数
# ============================================================================

def visualize_bbox(
    image: Image.Image,
    bbox: List[float],
    output_path: str
) -> None:
    """
    在图像上绘制边界框并保存。

    参数:
        image: PIL 图像对象
        bbox: [x1, y1, x2, y2] 相对坐标列表
        output_path: 保存可视化图像的路径
    """
    # 创建副本以避免修改原图
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)

    width, height = image.size

    # 将相对坐标转换为绝对像素坐标
    abs_x1 = int(bbox[0] * width)
    abs_y1 = int(bbox[1] * height)
    abs_x2 = int(bbox[2] * width)
    abs_y2 = int(bbox[3] * height)

    # 绘制红色矩形框，线宽 3px
    draw.rectangle(
        [abs_x1, abs_y1, abs_x2, abs_y2],
        outline="red",
        width=3
    )

    # 保存
    vis_image.save(output_path)
    print(f"  已保存可视化: {output_path}")


# ============================================================================
# 模块 E: 批量处理器
# ============================================================================

def process_image_folder(
    input_folder: str,
    model,
    tokenizer,
    temperature: float,
    visualize: bool = False,
    viz_dir: Optional[str] = None
) -> List[Dict]:
    """
    处理文件夹中的所有图像。

    参数:
        input_folder: 包含图像的文件夹路径
        model: 已加载的 Qwen3-VL 模型
        tokenizer: 已加载的分词器
        temperature: 采样温度
        visualize: 是否生成可视化
        viz_dir: 可视化图像保存目录

    返回:
        结果字典列表
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件夹: {input_folder}")

    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 查找所有图像
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"警告: 在 {input_folder} 中未找到图像")
        return []

    print(f"\n找到 {len(image_files)} 张待处理图像")
    print("=" * 60)

    results = []

    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 正在处理: {image_file.name}")

        result = {
            "image_name": image_file.name,
            "image_path": str(image_file.absolute()),
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "status": "failed",
            "error_msg": ""
        }

        try:
            # 加载图像
            image = Image.open(image_file)

            # 如有需要，转换为 RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 从模型获取标注
            print("  正在运行模型推理...")
            response = annotate_single_image(
                image,
                model,
                tokenizer,
                temperature=temperature
            )

            print(f"  模型回复: {response[:100]}...")

            # 解析边界框
            bbox = parse_bbox_from_text(response)

            if bbox is None:
                result["error_msg"] = "无法从回复中解析边界框"
                print(f"  错误: {result['error_msg']}")
                results.append(result)
                continue

            # 验证边界框
            is_valid, error_msg = validate_bbox(bbox)

            if not is_valid:
                result["error_msg"] = f"边界框无效: {error_msg}"
                print(f"  错误: {result['error_msg']}")
                results.append(result)
                continue

            # 成功！
            result["x1"] = f"{bbox[0]:.4f}"
            result["y1"] = f"{bbox[1]:.4f}"
            result["x2"] = f"{bbox[2]:.4f}"
            result["y2"] = f"{bbox[3]:.4f}"
            result["status"] = "success"

            print(f"  成功: bbox = [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")

            # 如需要，生成可视化
            if visualize and viz_dir:
                viz_path = Path(viz_dir) / image_file.name
                visualize_bbox(image, bbox, str(viz_path))

        except Exception as e:
            result["error_msg"] = f"异常: {str(e)}"
            print(f"  错误: {result['error_msg']}")

        results.append(result)

    return results


# ============================================================================
# 模块 F: CSV 输出管理器
# ============================================================================

def save_to_csv(results: List[Dict], output_path: str) -> None:
    """
    将标注结果保存到 CSV 文件。

    参数:
        results: 结果字典列表
        output_path: 输出 CSV 文件路径
    """
    if not results:
        print("没有结果需要保存。")
        return

    fieldnames = ["image_name", "image_path", "x1", "y1", "x2", "y2", "status", "error_msg"]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = len(results) - success_count

    print(f"\n{'=' * 60}")
    print(f"结果已保存至: {output_path}")
    print(f"总计: {len(results)} | 成功: {success_count} | 失败: {fail_count}")


def save_error_log(results: List[Dict], log_path: str) -> None:
    """
    保存失败标注的错误日志。

    参数:
        results: 结果字典列表
        log_path: 错误日志文件路径
    """
    failed_results = [r for r in results if r["status"] == "failed"]

    if not failed_results:
        print("没有错误需要记录。")
        return

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"皮肤病标注错误日志\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for result in failed_results:
            f.write(f"图像: {result['image_name']}\n")
            f.write(f"路径: {result['image_path']}\n")
            f.write(f"错误: {result['error_msg']}\n")
            f.write("-" * 40 + "\n\n")

    print(f"错误日志已保存至: {log_path}")


# ============================================================================
# 模块 G: 命令行接口
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。

    返回:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="使用 Qwen3-VL 为皮肤病图像标注边界框",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python annotate_skin_disease.py --input ./images --output annotations.csv
  python annotate_skin_disease.py -i ./images -o results.csv --temperature 0.5
  python annotate_skin_disease.py --input ./images --visualize --viz-dir ./results
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="包含皮肤病图像的输入文件夹"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="annotations.csv",
        help="输出 CSV 文件路径（默认: annotations.csv）"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        help="模型路径或 HuggingFace 标识符（默认: unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit）"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度，越低越确定性（默认: 0.7）"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="生成带边界框的可视化图像（默认: True）"
    )

    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="禁用可视化生成"
    )

    parser.add_argument(
        "--viz-dir",
        type=str,
        default="visualized",
        help="可视化输出目录（默认: visualized）"
    )

    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================

def main() -> int:
    """
    标注脚本的主入口。

    返回:
        退出码（0 表示成功，1 表示错误）
    """
    # 解析参数
    args = parse_arguments()

    print("=" * 60)
    print("皮肤病图像标注脚本")
    print("=" * 60)
    print(f"输入文件夹: {args.input}")
    print(f"输出 CSV: {args.output}")
    print(f"模型: {args.model}")
    print(f"温度参数: {args.temperature}")
    print(f"可视化: {args.visualize}")
    if args.visualize:
        print(f"可视化目录: {args.viz_dir}")
    print("=" * 60)

    # 如需要，创建可视化目录
    if args.visualize:
        viz_path = Path(args.viz_dir)
        viz_path.mkdir(parents=True, exist_ok=True)
        print(f"已创建可视化目录: {viz_path}")

    try:
        # 加载模型
        print("\n" + "=" * 60)
        model, tokenizer = load_model(args.model)

        # 处理图像
        results = process_image_folder(
            args.input,
            model,
            tokenizer,
            args.temperature,
            visualize=args.visualize,
            viz_dir=args.viz_dir if args.visualize else None
        )

        # 保存结果
        if results:
            save_to_csv(results, args.output)

            # 如有失败，保存错误日志
            if any(r["status"] == "failed" for r in results):
                save_error_log(results, "annotation_errors.log")

        print("\n" + "=" * 60)
        print("标注完成！")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        return 1
    except Exception as e:
        print(f"\n错误: 发生意外异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

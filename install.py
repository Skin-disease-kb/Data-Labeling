# -*- coding: utf-8 -*-
"""
Skin Disease Annotation Script - 一键安装脚本

基于官方 Unsloth Qwen3-VL 安装模板
自动检测环境并安装所有必需的依赖

使用方法:
    python install.py
"""

import os
import re
import subprocess
import sys


def run_command(cmd, description=""):
    """运行命令并显示输出"""
    print(f"\n{'=' * 60}")
    print(f"执行: {description or cmd}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"⚠️  警告: 命令执行返回非零状态码: {result.returncode}")

    return result.returncode == 0


def main():
    print("=" * 60)
    print("皮肤病标注脚本 - 依赖安装程序")
    print("=" * 60)
    print("\n基于官方 Unsloth Qwen3-VL 安装模板")
    print("将自动检测环境并安装所有必需依赖\n")

    # 检测是否在 Colab 环境
    colab_mode = "COLAB_" in "".join(os.environ.keys())

    if colab_mode:
        print("🟢 检测到 Google Colab 环境")
        print("使用 Colab 优化安装方式\n")
    else:
        print("🔵 检测到本地环境")
        print("使用标准安装方式\n")

    # ============================================================================
    # Colab 环境安装
    # ============================================================================
    if colab_mode:
        print("Step 1: 检测 PyTorch 版本...")
        try:
            import torch
            v = re.match(r"[0-9]{1,}\.[0-9]{1,}", str(torch.__version__)).group(0)
            print(f"PyTorch 版本: {torch.__version__}")
            print(f"主版本号: {v}")

            # 根据 PyTorch 版本选择 xformers
            xformers_version = {
                "2.9": "0.0.33.post1",
                "2.8": "0.0.32.post2",
            }.get(v, "0.0.29.post3")

            print(f"xformers 版本: {xformers_version}\n")

        except Exception as e:
            print(f"⚠️  无法检测 PyTorch 版本: {e}")
            print("使用默认 xformers 版本\n")
            xformers_version = "0.0.29.post3"

        print("Step 2: 安装核心依赖（无模式）...")
        run_command(
            f'pip install --no-deps bitsandbytes accelerate "{xformers_version}" peft trl triton cut_cross_entropy unsloth_zoo',
            "安装核心推理依赖"
        )

        print("\nStep 3: 安装文本处理和传输工具...")
        run_command(
            'pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer',
            "安装文本处理和传输工具"
        )

        print("\nStep 4: 安装 Unsloth...")
        run_command(
            'pip install --no-deps unsloth',
            "安装 Unsloth 核心库"
        )

        print("\nStep 5: 安装指定版本的 transformers 和 trl...")
        run_command(
            'pip install transformers==4.57.1',
            "安装 transformers 4.57.1"
        )
        run_command(
            'pip install --no-deps trl==0.22.2',
            "安装 trl 0.22.2"
        )

    # ============================================================================
    # 本地环境安装
    # ============================================================================
    else:
        print("Step 1: 安装 Unsloth（会自动安装大部分依赖）...")
        run_command(
            'pip install unsloth',
            "安装 Unsloth"
        )

        print("\nStep 2: 安装指定版本的依赖...")
        run_command(
            'pip install transformers==4.57.1',
            "安装 transformers 4.57.1"
        )
        run_command(
            'pip install --no-deps trl==0.22.2',
            "安装 trl 0.22.2"
        )

        print("\nStep 3: 安装图像处理依赖...")
        run_command(
            'pip install Pillow',
            "安装 Pillow"
        )

    # ============================================================================
    # 通用步骤（两种环境都需要）
    # ============================================================================
    print("\n" + "=" * 60)
    print("Step 4: 验证安装...")
    print("=" * 60)

    # 验证关键包
    packages_to_check = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("unsloth", "Unsloth"),
        ("PIL", "Pillow"),
    ]

    print("\n已安装的包:")
    for module_name, display_name in packages_to_check:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} (未安装)")

    # ============================================================================
    # 完成
    # ============================================================================
    print("\n" + "=" * 60)
    print("🎉 安装完成！")
    print("=" * 60)
    print("\n接下来可以运行标注脚本:")
    print("  python annotate_skin_disease.py --input <图像文件夹> --output annotations.csv")
    print("\n示例:")
    print("  python annotate_skin_disease.py --input ./images --output annotations.csv")
    print("\n如需帮助:")
    print("  python annotate_skin_disease.py --help")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  安装被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 安装过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

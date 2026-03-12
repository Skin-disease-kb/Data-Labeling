#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于原始皮肤病图像的 Qwen3-VL-8B LoRA 分类微调脚本。

数据目录结构:
data/
  ACK_光化性角化病/*.png
  DF_皮肤纤维瘤/*.jpg
  MEL_黑色素瘤/*.png
  SCC_鳞状细胞癌/*.png
  SEK_脂溢性角化病/*.png

训练目标:
- 输入 1 张皮肤病图像
- 仅输出 1 个固定标签: ACK / DF / MEL / SCC / SEK
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from datasets import Dataset, Image as HFImage
from PIL import Image

CACHE_ROOT = Path(os.environ.get("QWEN_DERM_CACHE_DIR", str(Path.cwd() / ".cache")))
TRITON_CACHE_DIR = Path(os.environ.get("TRITON_CACHE_DIR", str(CACHE_ROOT / "triton")))
TORCHINDUCTOR_CACHE_DIR = Path(
    os.environ.get("TORCHINDUCTOR_CACHE_DIR", str(CACHE_ROOT / "torchinductor"))
)

os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(TORCHINDUCTOR_CACHE_DIR))

TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TORCHINDUCTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

TRL_IMPORT_ERROR = None
UNSLOTH_IMPORT_ERROR = None

try:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
except Exception as exc:  # pragma: no cover - import guard for local env differences
    FastVisionModel = None
    UnslothVisionDataCollator = None
    UNSLOTH_IMPORT_ERROR = exc

from transformers import TrainerCallback

try:
    from trl import SFTConfig, SFTTrainer
except Exception as exc:  # pragma: no cover - import guard for local env differences
    SFTConfig = None
    SFTTrainer = None
    TRL_IMPORT_ERROR = exc


DEFAULT_MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
DEFAULT_FALLBACK_MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"
DEFAULT_DATA_DIR = Path(r"D:\qwen-derm\data")
DEFAULT_OUTPUT_DIR = Path(r"D:\qwen-derm\outputs\qwen3_vl_skin_cls_lora_raw")
LOW_VRAM_ERROR_SNIPPET = "Some modules are dispatched on the CPU or the disk"

VALID_LABELS = ["ACK", "DF", "MEL", "SCC", "SEK"]
CONFUSION_MATRIX_LABELS = VALID_LABELS + ["INVALID"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_MAP = {
    "ACK_光化性角化病": "ACK",
    "DF_皮肤纤维瘤": "DF",
    "MEL_黑色素瘤": "MEL",
    "SCC_鳞状细胞癌": "SCC",
    "SEK_脂溢性角化病": "SEK",
}
IGNORED_DIRECTORIES = {"SEK_脂溢性角化病（高精图片版）"}
CLASSIFICATION_INSTRUCTION = (
    "Classify the skin lesion in this image. Choose exactly one label from: "
    "ACK, DF, MEL, SCC, SEK. Respond with the label only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL raw-image LoRA classification trainer")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="原始图像数据目录")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="基础模型 ID 或路径")
    parser.add_argument(
        "--fallback-model-name",
        type=str,
        default="",
        help="8B 显存不足时使用的回退模型；留空则不回退",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="训练输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--max-length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--batch-size", type=int, default=1, help="单卡 batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument(
        "--oversample-floor-ratio",
        type=float,
        default=0.5,
        help="少数类过采样下限，相对训练集中最大类样本数的比例",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="仅用于 smoke test；设置后将覆盖完整 epoch 训练",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_counter(title: str, counts: Counter) -> None:
    print(f"\n{title}")
    for label in VALID_LABELS:
        print(f"  {label}: {counts.get(label, 0)}")


def discover_records(data_dir: Path) -> List[Dict]:
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    records: List[Dict] = []
    discovered_dirs = []
    ignored_dirs = []

    for child in sorted(data_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue

        if child.name in IGNORED_DIRECTORIES:
            ignored_dirs.append(child.name)
            continue

        label = LABEL_MAP.get(child.name)
        if label is None:
            print(f"警告: 跳过未识别目录 {child}")
            continue

        discovered_dirs.append(child.name)
        for image_path in sorted(child.iterdir(), key=lambda p: p.name):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            records.append(
                {
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "label": label,
                    "source_dir": child.name,
                }
            )

    if not records:
        raise RuntimeError(f"未在 {data_dir} 中发现可用图像")

    print("\n已发现类别目录:")
    for directory in discovered_dirs:
        print(f"  {directory}")
    if ignored_dirs:
        print("\n已忽略目录:")
        for directory in ignored_dirs:
            print(f"  {directory}")

    return records


def stratified_split(records: Sequence[Dict], val_ratio: float, seed: int) -> tuple[List[Dict], List[Dict]]:
    labels = [record["label"] for record in records]
    train_records, val_records = train_test_split(
        list(records),
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train_records, val_records


def oversample_training_records(
    train_records: Sequence[Dict],
    floor_ratio: float,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in train_records:
        grouped[record["label"]].append(dict(record))

    max_count = max(len(items) for items in grouped.values())
    floor_count = max(1, math.ceil(max_count * floor_ratio))

    oversampled: List[Dict] = []
    for label in VALID_LABELS:
        samples = grouped.get(label, [])
        if not samples:
            continue
        oversampled.extend(samples)
        if len(samples) < floor_count:
            need = floor_count - len(samples)
            for index in range(need):
                duplicate = dict(rng.choice(samples))
                duplicate["oversampled"] = True
                duplicate["oversample_index"] = index
                oversampled.append(duplicate)

    rng.shuffle(oversampled)
    return oversampled


def build_messages(example: Dict) -> Dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": CLASSIFICATION_INSTRUCTION},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["label"]}],
            },
        ]
    }


def build_hf_dataset(records: Sequence[Dict]) -> Dataset:
    dataset = Dataset.from_list(
        [{"image": record["image_path"], "label": record["label"]} for record in records]
    )
    dataset = dataset.cast_column("image", HFImage())
    dataset = dataset.map(build_messages, remove_columns=dataset.column_names)
    return dataset


def save_distribution_summary(
    output_dir: Path,
    train_records: Sequence[Dict],
    train_oversampled_records: Sequence[Dict],
    val_records: Sequence[Dict],
) -> None:
    summary = {
        "train_before_oversample": dict(Counter(record["label"] for record in train_records)),
        "train_after_oversample": dict(Counter(record["label"] for record in train_oversampled_records)),
        "val": dict(Counter(record["label"] for record in val_records)),
        "labels": VALID_LABELS,
    }
    save_json(output_dir / "split_summary.json", summary)


def load_model(model_name: str, max_length: int, fallback_model_name: str | None = None):
    def _load(target_model_name: str):
        print(f"\n加载模型: {target_model_name}")
        model, tokenizer = FastVisionModel.from_pretrained(
            target_model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            max_seq_length=max_length,
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_rslora=False,
            loftq_config=None,
            random_state=3407,
        )
        return model, tokenizer, target_model_name

    try:
        return _load(model_name)
    except ValueError as exc:
        if fallback_model_name and LOW_VRAM_ERROR_SNIPPET in str(exc):
            print(
                f"\n警告: {model_name} 在当前显存条件下无法装入，"
                f"回退到 {fallback_model_name} 继续执行。"
            )
            return _load(fallback_model_name)
        raise


def summarize_trainable_params(model) -> Dict[str, float]:
    total_params = 0
    trainable_params = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total_params += count
        if parameter.requires_grad:
            trainable_params += count

    ratio = trainable_params / total_params if total_params else 0.0
    summary = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_ratio": ratio,
    }
    print("\n参数摘要:")
    print(f"  total_params:      {summary['total_params']:,}")
    print(f"  trainable_params:  {summary['trainable_params']:,}")
    print(f"  trainable_ratio:   {summary['trainable_ratio']:.6f}")
    return summary


def decode_generated_text(outputs, inputs, tokenizer) -> str:
    output_ids = outputs[0]
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[input_length:] if output_ids.shape[0] > input_length else output_ids
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not response:
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return response


def normalize_prediction(text: str) -> str:
    cleaned = text.strip().upper()
    if cleaned in VALID_LABELS:
        return cleaned

    match = re.search(r"\b(ACK|DF|MEL|SCC|SEK)\b", cleaned)
    if match:
        return match.group(1)

    chinese_to_label = {
        "光化性角化病": "ACK",
        "皮肤纤维瘤": "DF",
        "黑色素瘤": "MEL",
        "鳞状细胞癌": "SCC",
        "脂溢性角化病": "SEK",
    }
    for chinese_name, label in chinese_to_label.items():
        if chinese_name in text:
            return label

    return "INVALID"


def predict_label(model, tokenizer, image_path: str) -> tuple[str, str]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": CLASSIFICATION_INSTRUCTION},
                ],
            }
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )

    raw_text = decode_generated_text(outputs, inputs, tokenizer)
    normalized = normalize_prediction(raw_text)
    return normalized, raw_text


def write_predictions_csv(path: Path, rows: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_name", "image_path", "label", "pred_label", "is_correct", "raw_output"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_confusion_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + CONFUSION_MATRIX_LABELS)
        for label, row in zip(CONFUSION_MATRIX_LABELS, matrix.tolist()):
            writer.writerow([label] + row)


def write_report_csv(path: Path, report_dict: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label in VALID_LABELS + ["macro avg", "weighted avg"]:
            if label not in report_dict:
                continue
            row = report_dict[label]
            writer.writerow(
                [
                    label,
                    row.get("precision", 0.0),
                    row.get("recall", 0.0),
                    row.get("f1-score", 0.0),
                    row.get("support", 0),
                ]
            )


def evaluate_model(
    model,
    tokenizer,
    records: Sequence[Dict],
    output_dir: Path,
    prefix: str,
) -> Dict[str, float]:
    ensure_dir(output_dir)
    FastVisionModel.for_inference(model)
    model.eval()

    y_true: List[str] = []
    y_pred: List[str] = []
    prediction_rows: List[Dict] = []

    print(f"\n开始评估: {prefix} (样本数={len(records)})")
    for index, record in enumerate(records, start=1):
        pred_label, raw_output = predict_label(model, tokenizer, record["image_path"])
        y_true.append(record["label"])
        y_pred.append(pred_label)
        prediction_rows.append(
            {
                "image_name": record["image_name"],
                "image_path": record["image_path"],
                "label": record["label"],
                "pred_label": pred_label,
                "is_correct": pred_label == record["label"],
                "raw_output": raw_output,
            }
        )
        if index % 20 == 0 or index == len(records):
            print(f"  已评估 {index}/{len(records)}")

    invalid_count = sum(pred == "INVALID" for pred in y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=VALID_LABELS, average="macro", zero_division=0)),
        "invalid_count": int(invalid_count),
        "invalid_rate": float(invalid_count / len(y_pred) if y_pred else 0.0),
        "total": len(y_true),
    }

    matrix = confusion_matrix(y_true, y_pred, labels=CONFUSION_MATRIX_LABELS)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=VALID_LABELS,
        output_dict=True,
        zero_division=0,
    )

    print(
        "  accuracy={accuracy:.4f}  macro_f1={macro_f1:.4f}  invalid_rate={invalid_rate:.4f}".format(
            **metrics
        )
    )

    prefix_dir = output_dir / prefix
    ensure_dir(prefix_dir)
    save_json(prefix_dir / "metrics.json", metrics)
    save_json(prefix_dir / "classification_report.json", report_dict)
    write_report_csv(prefix_dir / "classification_report.csv", report_dict)
    write_confusion_matrix_csv(prefix_dir / "confusion_matrix.csv", matrix)
    write_predictions_csv(prefix_dir / "predictions.csv", prediction_rows)
    return metrics


def save_adapter(model, tokenizer, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    ensure_dir(target_dir)
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


class EpochEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_records: Sequence[Dict], output_dir: Path):
        self.tokenizer = tokenizer
        self.eval_records = list(eval_records)
        self.output_dir = output_dir
        self.best_macro_f1 = -1.0
        self.best_epoch = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        epoch_number = state.epoch if state.epoch is not None else 0
        prefix = f"epoch_{int(round(epoch_number)):02d}"
        metrics = evaluate_model(model, self.tokenizer, self.eval_records, self.output_dir / "eval", prefix)
        if metrics["macro_f1"] > self.best_macro_f1:
            self.best_macro_f1 = metrics["macro_f1"]
            self.best_epoch = epoch_number
            save_adapter(model, self.tokenizer, self.output_dir / "best_adapter")
            save_json(
                self.output_dir / "best_adapter" / "best_metrics.json",
                {"epoch": epoch_number, **metrics},
            )

        state.log_history.append(
            {
                "epoch": epoch_number,
                "eval_accuracy": metrics["accuracy"],
                "eval_macro_f1": metrics["macro_f1"],
                "eval_invalid_rate": metrics["invalid_rate"],
            }
        )
        FastVisionModel.for_training(model)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.best_epoch is not None:
            save_json(
                self.output_dir / "best_summary.json",
                {"best_epoch": self.best_epoch, "best_macro_f1": self.best_macro_f1},
            )
        return control


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    if TRL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "无法导入 TRL 训练依赖。请先执行 `pip install -r requirements-full.txt`，"
            f"原始错误: {TRL_IMPORT_ERROR}"
        ) from TRL_IMPORT_ERROR
    if UNSLOTH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "无法导入 Unsloth 训练依赖。请先确认已安装 CUDA 版 PyTorch，并执行 "
            "`pip install -r requirements-full.txt`，"
            f"原始错误: {UNSLOTH_IMPORT_ERROR}"
        ) from UNSLOTH_IMPORT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU，本脚本仅支持 GPU 训练。")

    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型: {args.model_name}")
    if args.fallback_model_name:
        print(f"回退模型: {args.fallback_model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    save_json(args.output_dir / "label_mapping.json", LABEL_MAP)

    records = discover_records(args.data_dir)
    all_counts = Counter(record["label"] for record in records)
    print_counter("全部数据分布:", all_counts)

    train_records, val_records = stratified_split(records, args.val_ratio, args.seed)
    train_counts_before = Counter(record["label"] for record in train_records)
    val_counts = Counter(record["label"] for record in val_records)
    print_counter("训练集分布（过采样前）:", train_counts_before)
    print_counter("验证集分布:", val_counts)

    oversampled_train_records = oversample_training_records(
        train_records,
        floor_ratio=args.oversample_floor_ratio,
        seed=args.seed,
    )
    train_counts_after = Counter(record["label"] for record in oversampled_train_records)
    print_counter("训练集分布（过采样后）:", train_counts_after)
    save_distribution_summary(args.output_dir, train_records, oversampled_train_records, val_records)

    train_dataset = build_hf_dataset(oversampled_train_records)

    model, tokenizer, loaded_model_name = load_model(
        args.model_name,
        args.max_length,
        fallback_model_name=args.fallback_model_name or None,
    )
    param_summary = summarize_trainable_params(model)
    save_json(args.output_dir / "trainable_params.json", param_summary)
    save_json(
        args.output_dir / "model_resolution.json",
        {
            "requested_model_name": args.model_name,
            "fallback_model_name": args.fallback_model_name or None,
            "loaded_model_name": loaded_model_name,
        },
    )

    baseline_metrics = evaluate_model(model, tokenizer, val_records, args.output_dir / "eval", "baseline")
    save_json(args.output_dir / "baseline_summary.json", baseline_metrics)

    FastVisionModel.for_training(model)
    data_collator = UnslothVisionDataCollator(model, tokenizer)
    eval_callback = EpochEvalCallback(tokenizer, val_records, args.output_dir)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs if args.max_steps is None else None,
            max_steps=args.max_steps if args.max_steps is not None else -1,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=str(args.output_dir / "trainer"),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_length,
            save_strategy="no",
            eval_strategy="no",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
        ),
        callbacks=[eval_callback],
    )

    trainer.train()

    final_adapter_dir = args.output_dir / "final_adapter"
    save_adapter(model, tokenizer, final_adapter_dir)
    final_metrics = evaluate_model(model, tokenizer, val_records, args.output_dir / "eval", "final")
    save_json(
        args.output_dir / "training_summary.json",
        {
            "requested_model_name": args.model_name,
            "loaded_model_name": loaded_model_name,
            "baseline": baseline_metrics,
            "final": final_metrics,
            "requested_epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_length": args.max_length,
        },
    )

    print("\n训练完成。")
    print(f"最终 Adapter: {final_adapter_dir}")
    print(f"最佳 Adapter: {args.output_dir / 'best_adapter'}")
    print(f"最终指标文件: {args.output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()

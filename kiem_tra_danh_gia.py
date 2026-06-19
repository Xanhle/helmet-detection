from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def read_yaml(data_yaml: Path) -> tuple[dict[str, Any], Path]:
    """Đọc data.yaml và xác định thư mục gốc của bộ dữ liệu."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    root_value = config.get("path")
    if root_value:
        dataset_root = Path(root_value)
        if not dataset_root.is_absolute():
            dataset_root = (data_yaml.parent / dataset_root).resolve()
    else:
        dataset_root = data_yaml.parent.resolve()

    return config, dataset_root


def resolve_entry_path(value: str, dataset_root: Path, yaml_parent: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    candidate = (dataset_root / path).resolve()
    if candidate.exists():
        return candidate

    return (yaml_parent / path).resolve()


def collect_images(
    entry: str | list[str],
    dataset_root: Path,
    yaml_parent: Path,
) -> list[Path]:
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []

    for value in entries:
        source = resolve_entry_path(str(value), dataset_root, yaml_parent)

        if source.is_dir():
            images.extend(
                p.resolve()
                for p in source.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue

                image_path = Path(line)
                if not image_path.is_absolute():
                    image_path = (source.parent / image_path).resolve()

                if (
                    image_path.exists()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    images.append(image_path)
        elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(source)

    return sorted(set(images))


def get_split_images(data_yaml: Path, split: str) -> list[Path]:
    config, dataset_root = read_yaml(data_yaml)

    if split not in config or not config[split]:
        raise ValueError(
            f"Không tìm thấy split '{split}' trong {data_yaml.name}"
        )

    return collect_images(
        entry=config[split],
        dataset_root=dataset_root,
        yaml_parent=data_yaml.parent,
    )


def print_dataset_counts(data_yaml: Path) -> None:
    print("\n========== SỐ LƯỢNG ẢNH ==========")
    for split in ("train", "val", "test"):
        try:
            images = get_split_images(data_yaml, split)
            print(f"{split:>5}: {len(images)} ảnh")
        except ValueError:
            print(f"{split:>5}: Không được khai báo")


def create_subset_yaml(
    original_yaml: Path,
    split: str,
    limit: int,
    seed: int,
    output_dir: Path,
) -> Path:
    images = get_split_images(original_yaml, split)

    if limit <= 0:
        return original_yaml

    if len(images) < limit:
        raise ValueError(
            f"Tập '{split}' chỉ có {len(images)} ảnh, "
            f"không đủ để lấy {limit} ảnh."
        )

    rng = random.Random(seed)
    selected = rng.sample(images, limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    list_file = output_dir / f"{split}_{limit}_images.txt"
    list_file.write_text(
        "\n".join(str(p) for p in selected),
        encoding="utf-8",
    )

    config, dataset_root = read_yaml(original_yaml)
    config["path"] = str(dataset_root)
    config[split] = str(list_file.resolve())

    subset_yaml = output_dir / f"data_{split}_{limit}.yaml"
    with subset_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    return subset_yaml


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, float | str]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy trọng số mô hình: {model_path}"
        )

    print("\n" + "=" * 60)
    print(f"Đang đánh giá: {model_path}")
    print(f"Dữ liệu: {data_yaml}")
    print(f"Split: {split}")
    print("=" * 60)

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    inference_ms = float(metrics.speed.get("inference", 0.0))
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

    result: dict[str, float | str] = {
        "model": model_path.name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "inference_ms": inference_ms,
        "fps": fps,
    }

    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")
    print(f"mAP50         : {result['map50']:.4f}")
    print(f"mAP50-95      : {result['map50_95']:.4f}")
    print(f"Inference (ms): {inference_ms:.4f}")
    print(f"FPS ước tính  : {fps:.4f}")

    return result


def save_results_csv(
    rows: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    if not rows:
        return

    fieldnames = [
        "model",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "inference_ms",
        "fps",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nĐã lưu kết quả tại: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Đếm số ảnh và đánh giá YOLOv8n/YOLO26n "
            "trên cùng một tập dữ liệu."
        )
    )

    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model-v8", type=Path)
    parser.add_argument("--model-v26", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ket_qua_danh_gia.csv"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()

    print_dataset_counts(data_yaml)

    if args.count_only:
        print("\nĐã hoàn tất việc đếm ảnh.")
        return

    evaluation_yaml = data_yaml
    if args.limit > 0:
        evaluation_yaml = create_subset_yaml(
            original_yaml=data_yaml,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            output_dir=Path("evaluation_subset"),
        )
        print(
            f"\nĐã tạo tập đánh giá gồm {args.limit} ảnh: "
            f"{evaluation_yaml.resolve()}"
        )

    models = [
        ("YOLOv8n", args.model_v8),
        ("YOLO26n", args.model_v26),
    ]

    results: list[dict[str, float | str]] = []
    for display_name, model_path in models:
        if model_path is None:
            continue

        result = evaluate_model(
            model_path=model_path.resolve(),
            data_yaml=evaluation_yaml,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        result["model"] = display_name
        results.append(result)

    if not results:
        print(
            "\nBạn chưa truyền --model-v8 hoặc --model-v26. "
            "Chương trình chỉ đếm số ảnh."
        )
        return

    save_results_csv(results, args.output)


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# CẤU HÌNH
# =========================

OUTPUT_DIR = Path("comparison_yolo_versions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "comparison_yolov8n_2_datasets.png"


# =========================
# DỮ LIỆU SO SÁNH
# =========================

datasets = ["Dataset Robofolow", "Dataset Kaggle"]

metrics = {
    "Precision": [0.814, 0.916],
    "Recall": [0.829, 0.933],
    "mAP50": [0.870, 0.944],
    "mAP50-95": [0.519, 0.479],
}


# =========================
# VẼ BIỂU ĐỒ
# =========================

def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )


def main():
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=metric_name
        )
        add_value_labels(ax, bars)

    ax.set_title("Comparison of YOLOv8n on Two Datasets", fontsize=16)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(0, 1.05)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("Đã lưu biểu đồ tại:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
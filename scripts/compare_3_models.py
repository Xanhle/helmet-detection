import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================

# Nơi chứa kết quả train của từng mô hình
RUNS_DIR = Path("runs/detect")

# Nơi lưu kết quả so sánh 3 mô hình
OUTPUT_DIR = Path("comparison_dataset_500")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# THƯ MỤC KẾT QUẢ TỪNG MÔ HÌNH
# =========================
# Nếu folder của em khác tên, chỉ sửa ở đây.

MODEL_DIRS = {
    "YOLOv8n500": [
        RUNS_DIR / "yolov8n500" / "results",
        RUNS_DIR / "yolov8n500",
    ],
    "Faster R-CNN": [
        RUNS_DIR / "frcnn" / "results",
        RUNS_DIR / "frcnn" / "weights",
        RUNS_DIR / "frcnn",
    ],
    "SSD": [
        RUNS_DIR / "ssd" / "results",
        RUNS_DIR / "ssd" / "weights",
        RUNS_DIR / "ssd",
    ],
}


# =========================
# FILE MODEL ĐỂ TÍNH DUNG LƯỢNG
# =========================

MODEL_WEIGHT_CANDIDATES = {
    "YOLOv8n500": [
        RUNS_DIR / "yolov8n500" / "results" / "weights" / "best.pt",
        RUNS_DIR / "yolov8n500" / "weights" / "best.pt",
    ],
    "Faster R-CNN": [
        RUNS_DIR / "frcnn" / "results" / "frcnn_final.pth",
        RUNS_DIR / "frcnn" / "weights" / "frcnn_final.pth",
        RUNS_DIR / "frcnn" / "frcnn_final.pth",
    ],
    "SSD": [
        RUNS_DIR / "ssd" / "results" / "ssd_final.pth",
        RUNS_DIR / "ssd" / "weights" / "ssd_final.pth",
        RUNS_DIR / "ssd" / "ssd_final.pth",
    ],
}


# =========================
# FILE CSV CÓ THỂ CÓ TRONG TỪNG FOLDER
# =========================

METRIC_CSV_CANDIDATES = [
    "metrics_with_accuracy_mae.csv",
    "metrics_summary.csv",
    "metrics.csv",
    "results.csv",
]


# =========================
# THÔNG SỐ TỐC ĐỘ TRIỂN KHAI
# =========================
# Nếu em có số đo mới thì sửa ở đây.

DEPLOY_METRICS = {
    "YOLOv8n500": {
        "Inference time (ms/img)": 5.95,
        "FPS": 167.9
    },
    "Faster R-CNN": {
        "Inference time (ms/img)": 91.11,
        "FPS": 11.0
    },
    "SSD": {
        "Inference time (ms/img)": 2.48,
        "FPS": 402.5
    },
}


# =========================
# HÀM PHỤ
# =========================

def find_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def find_metric_csv(model_name):
    model_dir = find_existing_path(MODEL_DIRS[model_name])

    if model_dir is None:
        return None

    for csv_name in METRIC_CSV_CANDIDATES:
        csv_path = model_dir / csv_name
        if csv_path.exists():
            return csv_path

    return None


def find_weight_file(model_name):
    weight_path = find_existing_path(MODEL_WEIGHT_CANDIDATES[model_name])
    return weight_path


def get_model_size_mb(model_name):
    weight_path = find_weight_file(model_name)

    if weight_path is None:
        return np.nan

    return weight_path.stat().st_size / (1024 * 1024)


def clean_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_first_existing_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def to_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def read_best_row_from_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = clean_columns(df)

    # Ép các cột có thể là số về numeric
    for col in df.columns:
        if col.lower() != "model":
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Ưu tiên chọn epoch tốt nhất theo mAP50 nếu file có nhiều dòng
    map50_col = get_first_existing_col(
        df,
        [
            "mAP@0.5",
            "map50",
            "mAP50",
            "metrics/mAP50(B)",
            "metrics/mAP50",
        ]
    )

    if map50_col is not None:
        numeric_map50 = pd.to_numeric(df[map50_col], errors="coerce")

        if numeric_map50.notna().any():
            best_idx = numeric_map50.idxmax()
            return df.loc[best_idx]

    # Nếu không có mAP50 thì lấy dòng cuối
    return df.iloc[-1]


def extract_metric(row, candidates):
    for col in candidates:
        if col in row.index:
            return to_float(row[col])
    return np.nan


def extract_model_metrics(model_name):
    csv_path = find_metric_csv(model_name)

    if csv_path is None:
        print(f"Không tìm thấy file metrics cho mô hình: {model_name}")
        return {
            "Model": model_name,
            "Best epoch": np.nan,
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1-score": np.nan,
            "MAE": np.nan,
            "mAP@0.5": np.nan,
            "mAP@0.5:0.95": np.nan,
            "Source CSV": ""
        }

    row = read_best_row_from_csv(csv_path)

    best_epoch = extract_metric(
        row,
        [
            "Best epoch",
            "best_epoch",
            "epoch",
            "Epoch"
        ]
    )

    accuracy = extract_metric(
        row,
        [
            "Accuracy",
            "accuracy",
            "Acc",
            "acc"
        ]
    )

    precision = extract_metric(
        row,
        [
            "Precision",
            "precision",
            "Box(P",
            "Box(P)",
            "metrics/precision(B)",
            "metrics/precision"
        ]
    )

    recall = extract_metric(
        row,
        [
            "Recall",
            "recall",
            "R",
            "Box(R)",
            "metrics/recall(B)",
            "metrics/recall"
        ]
    )

    f1 = extract_metric(
        row,
        [
            "F1-score",
            "F1",
            "f1",
            "f1_score",
            "F1 Score"
        ]
    )

    mae = extract_metric(
        row,
        [
            "MAE",
            "mae",
            "Mean Absolute Error"
        ]
    )

    map50 = extract_metric(
        row,
        [
            "mAP@0.5",
            "map50",
            "mAP50",
            "metrics/mAP50(B)",
            "metrics/mAP50"
        ]
    )

    map5095 = extract_metric(
        row,
        [
            "mAP@0.5:0.95",
            "mAP50-95",
            "map50_95",
            "mAP@0.5:0.95(B)",
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95"
        ]
    )

    return {
        "Model": model_name,
        "Best epoch": best_epoch,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "MAE": mae,
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095,
        "Source CSV": str(csv_path).replace("\\", "/")
    }


def add_value_labels(ax, bars, fmt="{:.3f}", y_offset_ratio=0.01):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min if y_max > y_min else 1.0
    offset = y_range * y_offset_ratio

    for bar in bars:
        height = bar.get_height()

        if pd.isna(height):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9
        )


def save_single_bar_chart(df, model_col, value_col, title, ylabel, output_path):
    if value_col not in df.columns:
        return

    valid_df = df[[model_col, value_col]].dropna()

    if valid_df.empty:
        print(f"Bỏ qua {output_path.name} vì không có dữ liệu.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(valid_df[model_col], valid_df[value_col])

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    add_value_labels(ax, bars)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu: {output_path}")


def save_grouped_bar_chart(df, model_col, metric_cols, title, ylabel, output_path):
    existing_cols = []

    for col in metric_cols:
        if col in df.columns and df[col].notna().any():
            existing_cols.append(col)

    if not existing_cols:
        print(f"Bỏ qua {output_path.name} vì không có dữ liệu.")
        return

    plot_df = df[[model_col] + existing_cols].copy()

    x = np.arange(len(plot_df))
    width = 0.8 / len(existing_cols)

    fig, ax = plt.subplots(figsize=(14, 7))

    all_bars = []

    for i, metric in enumerate(existing_cols):
        values = plot_df[metric].values

        bars = ax.bar(
            x + i * width - (len(existing_cols) - 1) * width / 2,
            values,
            width,
            label=metric
        )

        all_bars.append(bars)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[model_col])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in all_bars:
        add_value_labels(ax, bars)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu: {output_path}")


# =========================
# MAIN
# =========================

def main():
    rows = []

    for model_name in MODEL_DIRS.keys():
        print(f"Đang đọc kết quả mô hình: {model_name}")

        row = extract_model_metrics(model_name)

        row["Model size (MB)"] = get_model_size_mb(model_name)
        row["Inference time (ms/img)"] = DEPLOY_METRICS.get(
            model_name, {}
        ).get("Inference time (ms/img)", np.nan)
        row["FPS"] = DEPLOY_METRICS.get(
            model_name, {}
        ).get("FPS", np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sắp xếp thứ tự cột
    column_order = [
        "Model",
        "Best epoch",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "MAE",
        "mAP@0.5",
        "mAP@0.5:0.95",
        "Model size (MB)",
        "Inference time (ms/img)",
        "FPS",
        "Source CSV"
    ]

    df = df[[c for c in column_order if c in df.columns]]

    # Lưu bảng tổng hợp
    summary_csv = OUTPUT_DIR / "metrics_summary_3models.csv"
    full_csv = OUTPUT_DIR / "model_comparison_summary_full.csv"

    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    df.to_csv(full_csv, index=False, encoding="utf-8-sig")

    print("\n====================================")
    print("BẢNG SO SÁNH 3 MÔ HÌNH")
    print("====================================")
    print(df.to_string(index=False))

    print("\nĐã lưu:")
    print("-", summary_csv)
    print("-", full_csv)

    # Biểu đồ nhóm chỉ số chính
    save_grouped_bar_chart(
        df,
        "Model",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "mAP@0.5",
            "mAP@0.5:0.95"
        ],
        "Comparison of Main Metrics",
        "Score",
        OUTPUT_DIR / "comparison_main_metrics.png"
    )

    # Biểu đồ riêng từng chỉ số
    single_metric_configs = [
        ("Accuracy", "Comparison of Accuracy", "Score", "comparison_accuracy.png"),
        ("Precision", "Comparison of Precision", "Score", "comparison_precision.png"),
        ("Recall", "Comparison of Recall", "Score", "comparison_recall.png"),
        ("F1-score", "Comparison of F1-score", "Score", "comparison_f1.png"),
        ("MAE", "Comparison of MAE", "Error", "comparison_mae.png"),
        ("mAP@0.5", "Comparison of mAP@0.5", "Score", "comparison_map50.png"),
        ("mAP@0.5:0.95", "Comparison of mAP@0.5:0.95", "Score", "comparison_map50_95.png"),
        ("Model size (MB)", "Comparison of Model Size", "MB", "comparison_model_size.png"),
        ("Inference time (ms/img)", "Comparison of Inference Time", "ms/img", "comparison_inference_time.png"),
        ("FPS", "Comparison of FPS", "FPS", "comparison_fps.png"),
    ]

    for metric, title, ylabel, filename in single_metric_configs:
        save_single_bar_chart(
            df,
            "Model",
            metric,
            title,
            ylabel,
            OUTPUT_DIR / filename
        )

    print("\n====================================")
    print("ĐÃ TẠO XONG BIỂU ĐỒ SO SÁNH 3 MÔ HÌNH")
    print("Folder lưu kết quả:", OUTPUT_DIR)
    print("====================================")


if __name__ == "__main__":
    main()
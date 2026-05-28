import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# CẤU HÌNH
# =========================

RUNS_DIR = Path("runs/detect")

# Folder lưu riêng kết quả so sánh YOLOv8n và YOLO26n
OUTPUT_DIR = Path("comparison_yolo_versions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# FOLDER KẾT QUẢ TRAIN
# =========================
# Nếu em muốn so sánh YOLOv8n500 thì đổi YOLOv8n sang:
# RUNS_DIR / "yolov8n500" / "results"

MODEL_RUN_DIRS = {
    "YOLOv8n": [
        RUNS_DIR / "DoAn_Helmet_YOLOv8n_Finetune_20ep_640",
        RUNS_DIR / "DoAn_Helmet_nano-2",
        RUNS_DIR / "yolov8n500" / "results",
        RUNS_DIR / "yolov8n500",
    ],
    "YOLO26n": [
        RUNS_DIR / "DoAn_Helmet_YOLO26n_30ep_640_b1",
    ],
}


# =========================
# FILE WEIGHT ĐỂ TÍNH DUNG LƯỢNG MODEL
# =========================

MODEL_WEIGHT_CANDIDATES = {
    "YOLOv8n": [
        RUNS_DIR / "DoAn_Helmet_YOLOv8n_Finetune_20ep_640" / "weights" / "best.pt",
        RUNS_DIR / "DoAn_Helmet_nano-2" / "weights" / "best.pt",
        RUNS_DIR / "yolov8n500" / "results" / "weights" / "best.pt",
        RUNS_DIR / "yolov8n500" / "weights" / "best.pt",
    ],
    "YOLO26n": [
        RUNS_DIR / "DoAn_Helmet_YOLO26n_30ep_640_b1" / "weights" / "best.pt",
    ],
}


# =========================
# HÀM PHỤ
# =========================

def find_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def find_results_csv(model_name):
    run_dir = find_existing_path(MODEL_RUN_DIRS[model_name])

    if run_dir is None:
        return None

    candidates = [
        run_dir / "results.csv",
        run_dir / "metrics_summary.csv",
        run_dir / "metrics_with_accuracy_mae.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def find_weight_file(model_name):
    return find_existing_path(MODEL_WEIGHT_CANDIDATES[model_name])


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


def read_best_row(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = clean_columns(df)

    # Chuẩn hóa numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    map50_col = get_first_existing_col(
        df,
        [
            "metrics/mAP50(B)",
            "metrics/mAP50",
            "mAP50",
            "map50",
            "mAP@0.5",
        ]
    )

    if map50_col is not None:
        numeric_map50 = pd.to_numeric(df[map50_col], errors="coerce")

        if numeric_map50.notna().any():
            best_idx = numeric_map50.idxmax()
            return df.loc[best_idx], df

    return df.iloc[-1], df


def extract_metric(row, candidates):
    for col in candidates:
        if col in row.index:
            return to_float(row[col])

    return np.nan


def extract_yolo_metrics(model_name):
    csv_path = find_results_csv(model_name)

    if csv_path is None:
        print(f"Không tìm thấy results.csv cho {model_name}")
        return {
            "Model": model_name,
            "Best epoch": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "mAP50": np.nan,
            "mAP50-95": np.nan,
            "Model size (MB)": get_model_size_mb(model_name),
            "Source CSV": "",
            "Weight file": ""
        }

    best_row, full_df = read_best_row(csv_path)

    best_epoch = extract_metric(
        best_row,
        [
            "epoch",
            "Epoch",
            "Best epoch",
        ]
    )

    precision = extract_metric(
        best_row,
        [
            "metrics/precision(B)",
            "metrics/precision",
            "Precision",
            "precision",
            "Box(P)",
            "Box(P",
        ]
    )

    recall = extract_metric(
        best_row,
        [
            "metrics/recall(B)",
            "metrics/recall",
            "Recall",
            "recall",
            "Box(R)",
            "R",
        ]
    )

    map50 = extract_metric(
        best_row,
        [
            "metrics/mAP50(B)",
            "metrics/mAP50",
            "mAP50",
            "map50",
            "mAP@0.5",
        ]
    )

    map5095 = extract_metric(
        best_row,
        [
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95",
            "mAP50-95",
            "map50_95",
            "mAP@0.5:0.95",
        ]
    )

    weight_file = find_weight_file(model_name)

    return {
        "Model": model_name,
        "Best epoch": best_epoch,
        "Precision": precision,
        "Recall": recall,
        "mAP50": map50,
        "mAP50-95": map5095,
        "Model size (MB)": get_model_size_mb(model_name),
        "Source CSV": str(csv_path).replace("\\", "/"),
        "Weight file": str(weight_file).replace("\\", "/") if weight_file else ""
    }


def add_value_labels(ax, bars, fmt="{:.3f}"):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min if y_max > y_min else 1.0
    offset = y_range * 0.015

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
            fontsize=10
        )


def save_grouped_bar_chart(df, metric_cols, output_path):
    model_col = "Model"
    existing_cols = [c for c in metric_cols if c in df.columns and df[c].notna().any()]

    if not existing_cols:
        print("Không có dữ liệu để vẽ biểu đồ nhóm.")
        return

    plot_df = df[[model_col] + existing_cols].copy()

    x = np.arange(len(plot_df))
    width = 0.8 / len(existing_cols)

    fig, ax = plt.subplots(figsize=(12, 7))

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

    ax.set_title("Comparison of YOLOv8n and YOLO26n", fontsize=16)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[model_col])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for bars in all_bars:
        add_value_labels(ax, bars)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu: {output_path}")


def save_single_metric_chart(df, metric, output_path):
    if metric not in df.columns:
        return

    valid_df = df[["Model", metric]].dropna()

    if valid_df.empty:
        print(f"Bỏ qua {metric} vì không có dữ liệu.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(valid_df["Model"], valid_df[metric])

    ax.set_title(f"Comparison of {metric}", fontsize=16)
    ax.set_ylabel(metric, fontsize=12)

    if metric not in ["Model size (MB)"]:
        ax.set_ylim(0, 1.05)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    add_value_labels(ax, bars)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu: {output_path}")


def save_difference_table(df):
    if len(df) < 2:
        return None

    yolo8 = df[df["Model"] == "YOLOv8n"].iloc[0]
    yolo26 = df[df["Model"] == "YOLO26n"].iloc[0]

    rows = []

    for metric in ["Precision", "Recall", "mAP50", "mAP50-95", "Model size (MB)"]:
        diff = yolo26[metric] - yolo8[metric]

        rows.append({
            "Metric": metric,
            "YOLOv8n": yolo8[metric],
            "YOLO26n": yolo26[metric],
            "Difference YOLO26n - YOLOv8n": diff
        })

    diff_df = pd.DataFrame(rows)

    diff_csv = OUTPUT_DIR / "yolo8_vs_yolo26_difference.csv"
    diff_df.to_csv(diff_csv, index=False, encoding="utf-8-sig")

    print(f"Đã lưu bảng chênh lệch: {diff_csv}")

    return diff_df


# =========================
# MAIN
# =========================

def main():
    rows = []

    for model_name in ["YOLOv8n", "YOLO26n"]:
        print(f"Đang đọc kết quả: {model_name}")
        rows.append(extract_yolo_metrics(model_name))

    df = pd.DataFrame(rows)

    output_csv = OUTPUT_DIR / "yolo8_vs_yolo26_summary.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n====================================")
    print("BẢNG SO SÁNH YOLOv8n VÀ YOLO26n")
    print("====================================")
    print(df.to_string(index=False))

    print("\nĐã lưu bảng tổng hợp:")
    print(output_csv)

    diff_df = save_difference_table(df)

    if diff_df is not None:
        print("\n====================================")
        print("BẢNG CHÊNH LỆCH")
        print("====================================")
        print(diff_df.to_string(index=False))

    save_grouped_bar_chart(
        df,
        ["Precision", "Recall", "mAP50", "mAP50-95"],
        OUTPUT_DIR / "comparison_yolo8_yolo26_main_metrics.png"
    )

    save_single_metric_chart(
        df,
        "Precision",
        OUTPUT_DIR / "comparison_precision.png"
    )

    save_single_metric_chart(
        df,
        "Recall",
        OUTPUT_DIR / "comparison_recall.png"
    )

    save_single_metric_chart(
        df,
        "mAP50",
        OUTPUT_DIR / "comparison_map50.png"
    )

    save_single_metric_chart(
        df,
        "mAP50-95",
        OUTPUT_DIR / "comparison_map50_95.png"
    )

    save_single_metric_chart(
        df,
        "Model size (MB)",
        OUTPUT_DIR / "comparison_model_size.png"
    )

    print("\n====================================")
    print("ĐÃ SO SÁNH XONG YOLOv8n VÀ YOLO26n")
    print("Folder lưu kết quả:", OUTPUT_DIR)
    print("====================================")


if __name__ == "__main__":
    main()
import os
import json
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# CẤU HÌNH FASTER R-CNN
# =========================
TRAIN_JSON = "frcnn/annotations_train.json"
VAL_JSON = "frcnn/annotations_valid.json"

TRAIN_IMG_DIR = "data/train/images"
VAL_IMG_DIR = "data/valid/images"

OUT_DIR = "frcnn/weights"

# JSON của bạn hiện có 500 ảnh train, 100 ảnh valid
EPOCHS = 5
BATCH_SIZE = 2

# None = dùng toàn bộ ảnh trong JSON
TRAIN_LIMIT = None
VAL_LIMIT = None

LR = 0.0005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

CONF_THRES = 0.25
IOU_THRES = 0.50

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# DATASET COCO
# =========================
class CocoDetectionDataset(Dataset):
    def __init__(self, images_dir, ann_json, limit=None, shuffle=True, require_boxes=True):
        self.images_dir = Path(images_dir)
        self.ann_json = Path(ann_json)

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.images_dir}")

        if not self.ann_json.exists():
            raise FileNotFoundError(f"Không tìm thấy JSON: {self.ann_json}")

        with open(self.ann_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = sorted(data["categories"], key=lambda x: x["id"])
        self.class_names = [c["name"] for c in categories]

        # Torchvision dùng label 0 cho background, class thật bắt đầu từ 1
        self.cat_id_to_label = {c["id"]: i + 1 for i, c in enumerate(categories)}

        anns_by_img = {}

        for ann in data["annotations"]:
            img_id = ann["image_id"]
            x, y, bw, bh = ann["bbox"]

            if bw <= 1 or bh <= 1:
                continue

            cat_id = ann["category_id"]

            if cat_id not in self.cat_id_to_label:
                continue

            label = self.cat_id_to_label[cat_id]

            anns_by_img.setdefault(img_id, []).append({
                "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                "label": int(label)
            })

        image_files = []

        for ext in [
            "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
            "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP"
        ]:
            image_files.extend(self.images_dir.rglob(ext))

        name_to_path = {p.name: p for p in image_files}
        stem_to_path = {p.stem: p for p in image_files}

        items = []

        for img in data["images"]:
            img_id = img["id"]
            file_name = Path(img["file_name"]).name

            img_path = self.images_dir / file_name

            if not img_path.exists():
                img_path = name_to_path.get(file_name)

            if img_path is None or not img_path.exists():
                img_path = stem_to_path.get(Path(file_name).stem)

            if img_path is None or not img_path.exists():
                continue

            anns = anns_by_img.get(img_id, [])

            if require_boxes and len(anns) == 0:
                continue

            items.append({
                "id": img_id,
                "file_name": file_name,
                "path": str(img_path),
                "anns": anns
            })

        if shuffle:
            random.shuffle(items)

        if limit is not None:
            items = items[:limit]

        self.items = items

        print("\n--- Dataset ---")
        print("JSON:", self.ann_json)
        print("Images dir:", self.images_dir)
        print("Số ảnh dùng:", len(self.items))
        print("Classes:", self.class_names)

        if len(self.items) == 0:
            raise ValueError("Dataset rỗng. Kiểm tra JSON và thư mục ảnh.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        img = cv2.imread(item["path"])

        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {item['path']}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes = []
        labels = []

        for ann in item["anns"]:
            x1, y1, x2, y2 = ann["bbox"]

            x1 = max(0, min(float(x1), w - 1))
            y1 = max(0, min(float(y1), h - 1))
            x2 = max(0, min(float(x2), w - 1))
            y2 = max(0, min(float(y2), h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["label"]))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        image = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if len(boxes) > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([item["id"]], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =========================
# MODEL FASTER R-CNN
# =========================
def build_model(num_classes):
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

        print("\nĐang tạo Faster R-CNN pretrained COCO...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)

    except Exception as e:
        print("\nKhông dùng được pretrained COCO, chuyển sang weights=None.")
        print("Lý do:", e)
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# =========================
# EVALUATION
# =========================
def collect_predictions(model, loader, device):
    model.eval()
    records = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                records.append({
                    "gt_boxes": target["boxes"].cpu(),
                    "gt_labels": target["labels"].cpu(),
                    "pred_boxes": output["boxes"].cpu(),
                    "pred_labels": output["labels"].cpu(),
                    "pred_scores": output["scores"].cpu()
                })

    return records


def count_tp_fp_fn(records, cls_id, conf_thres, iou_thres):
    tp = 0
    fp = 0
    fn = 0

    for rec in records:
        gt_boxes = rec["gt_boxes"]
        gt_labels = rec["gt_labels"]

        pred_boxes = rec["pred_boxes"]
        pred_labels = rec["pred_labels"]
        pred_scores = rec["pred_scores"]

        gt_mask = gt_labels == cls_id
        gt_cls_boxes = gt_boxes[gt_mask]

        pred_mask = (pred_labels == cls_id) & (pred_scores >= conf_thres)
        pred_cls_boxes = pred_boxes[pred_mask]
        pred_cls_scores = pred_scores[pred_mask]

        if len(pred_cls_boxes) > 0:
            order = torch.argsort(pred_cls_scores, descending=True)
            pred_cls_boxes = pred_cls_boxes[order]

        matched = torch.zeros((len(gt_cls_boxes),), dtype=torch.bool)

        for pbox in pred_cls_boxes:
            if len(gt_cls_boxes) == 0:
                fp += 1
                continue

            ious = box_iou(pbox.unsqueeze(0), gt_cls_boxes)[0]
            best_iou, best_idx = torch.max(ious, dim=0)

            if best_iou >= iou_thres and not matched[best_idx]:
                tp += 1
                matched[best_idx] = True
            else:
                fp += 1

        fn += int((~matched).sum().item())

    return tp, fp, fn
def compute_metrics(records, num_classes, conf_thres, iou_thres):
    thresholds = np.linspace(0.0, 0.95, 50)

    precisions = []
    recalls = []
    f1s = []
    aps = []

    p_curves = []
    r_curves = []
    f_curves = []

    for cls_id in range(1, num_classes):
        tp, fp, fn = count_tp_fp_fn(records, cls_id, conf_thres, iou_thres)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        pc = []
        rc = []
        fc = []

        for th in thresholds:
            tp_t, fp_t, fn_t = count_tp_fp_fn(records, cls_id, th, iou_thres)

            p = tp_t / (tp_t + fp_t + 1e-9)
            r = tp_t / (tp_t + fn_t + 1e-9)
            f = 2 * p * r / (p + r + 1e-9)

            pc.append(p)
            rc.append(r)
            fc.append(f)

        pc = np.array(pc)
        rc = np.array(rc)
        fc = np.array(fc)

        order = np.argsort(rc)
        ap = np.trapz(pc[order], rc[order])
        ap = max(0.0, min(1.0, float(ap)))

        aps.append(ap)
        p_curves.append(pc)
        r_curves.append(rc)
        f_curves.append(fc)

    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "map50": float(np.mean(aps)) if aps else 0.0,
        "thresholds": thresholds,
        "p_curve": np.mean(p_curves, axis=0) if p_curves else np.zeros_like(thresholds),
        "r_curve": np.mean(r_curves, axis=0) if r_curves else np.zeros_like(thresholds),
        "f_curve": np.mean(f_curves, axis=0) if f_curves else np.zeros_like(thresholds)
    }


def build_confusion_matrix(records, num_classes, conf_thres, iou_thres):
    # Hàng: Predicted, cột: True, 0 là background.
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for rec in records:
        gt_boxes = rec["gt_boxes"]
        gt_labels = rec["gt_labels"]

        pred_boxes = rec["pred_boxes"]
        pred_labels = rec["pred_labels"]
        pred_scores = rec["pred_scores"]

        keep = pred_scores >= conf_thres
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        if len(pred_boxes) > 0:
            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]

        matched_gt = torch.zeros((len(gt_boxes),), dtype=torch.bool)

        for pbox, plabel in zip(pred_boxes, pred_labels):
            plabel = int(plabel.item())

            if len(gt_boxes) == 0:
                cm[plabel, 0] += 1
                continue

            ious = box_iou(pbox.unsqueeze(0), gt_boxes)[0]
            best_iou, best_idx = torch.max(ious, dim=0)

            if best_iou >= iou_thres and not matched_gt[best_idx]:
                true_label = int(gt_labels[best_idx].item())
                cm[plabel, true_label] += 1
                matched_gt[best_idx] = True
            else:
                cm[plabel, 0] += 1

        for idx, matched in enumerate(matched_gt):
            if not matched:
                true_label = int(gt_labels[idx].item())
                cm[0, true_label] += 1

    return cm


# =========================
# PLOT
# =========================
def plot_confusion_matrix(cm, names, save_path, title, normalize=False):
    if normalize:
        cm_plot = cm.astype(np.float32) / (cm.sum(axis=0, keepdims=True) + 1e-9)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(names))
    plt.xticks(ticks, names, rotation=45, ha="right")
    plt.yticks(ticks, names)

    thresh = cm_plot.max() / 2.0 if cm_plot.max() > 0 else 0.5

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            value = cm_plot[i, j]
            plt.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > thresh else "black"
            )

    plt.ylabel("Predicted")
    plt.xlabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_results(history, save_path):
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    plots = [
        ("train/total_loss", "total_loss"),
        ("train/bbox_loss", "bbox_loss"),
        ("train/cls_loss", "cls_loss"),
        ("metrics/precision(B)", "precision"),
        ("metrics/recall(B)", "recall"),
        ("metrics/F1(B)", "f1"),
        ("metrics/mAP50(B)", "map50"),
        ("lr", "lr")
    ]

    for ax, (title, key) in zip(axes.ravel(), plots):
        y = [h.get(key, 0) for h in history]
        ax.plot(epochs, y, marker="o", label="results")
        ax.set_title(title + " - Faster R-CNN")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_curves(metrics, out_dir):
    thresholds = metrics["thresholds"]
    p_curve = metrics["p_curve"]
    r_curve = metrics["r_curve"]
    f_curve = metrics["f_curve"]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, p_curve, linewidth=3, label="all classes")
    plt.title("Precision-Confidence Curve - Faster R-CNN")
    plt.xlabel("Confidence")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "BoxP_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, r_curve, linewidth=3, label="all classes")
    plt.title("Recall-Confidence Curve - Faster R-CNN")
    plt.xlabel("Confidence")
    plt.ylabel("Recall")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "BoxR_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f_curve, linewidth=3, label="all classes")
    plt.title("F1-Confidence Curve - Faster R-CNN")
    plt.xlabel("Confidence")
    plt.ylabel("F1")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "BoxF1_curve.png"), dpi=200)
    plt.close()

    order = np.argsort(r_curve)

    plt.figure(figsize=(8, 6))
    plt.plot(
        r_curve[order],
        p_curve[order],
        linewidth=3,
        label=f"all classes {metrics['map50']:.3f} mAP@0.5"
    )
    plt.title("Precision-Recall Curve - Faster R-CNN")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "BoxPR_curve.png"), dpi=200)
    plt.close()


def save_metrics_csv(history, out_dir):
    csv_path = os.path.join(out_dir, "metrics.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "total_loss",
                "bbox_loss",
                "cls_loss",
                "precision",
                "recall",
                "f1",
                "map50",
                "lr"
            ]
        )

        writer.writeheader()

        for row in history:
            writer.writerow(row)

    return csv_path


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = CocoDetectionDataset(
        TRAIN_IMG_DIR,
        TRAIN_JSON,
        limit=TRAIN_LIMIT,
        shuffle=True,
        require_boxes=True
    )

    val_dataset = CocoDetectionDataset(
        VAL_IMG_DIR,
        VAL_JSON,
        limit=VAL_LIMIT,
        shuffle=False,
        require_boxes=False
    )

    num_classes = len(train_dataset.class_names) + 1
    names = ["background"] + train_dataset.class_names

    print("\nSố lớp Faster R-CNN gồm background:", num_classes)
    print("Tên lớp:", names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = build_model(num_classes).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss_sum = 0.0
        bbox_loss_sum = 0.0
        cls_loss_sum = 0.0
        num_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for images, targets in loop:
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss = float(losses.item())
            bbox_loss = float(loss_dict.get("loss_box_reg", torch.tensor(0.0)).item())
            cls_loss = float(loss_dict.get("loss_classifier", torch.tensor(0.0)).item())

            total_loss_sum += total_loss
            bbox_loss_sum += bbox_loss
            cls_loss_sum += cls_loss
            num_batches += 1

            loop.set_postfix({
                "loss": f"{total_loss:.4f}",
                "bbox": f"{bbox_loss:.4f}",
                "cls": f"{cls_loss:.4f}"
            })

        scheduler.step()

        records = collect_predictions(model, val_loader, device)
        metrics = compute_metrics(records, num_classes, CONF_THRES, IOU_THRES)

        row = {
            "epoch": epoch,
            "total_loss": total_loss_sum / max(num_batches, 1),
            "bbox_loss": bbox_loss_sum / max(num_batches, 1),
            "cls_loss": cls_loss_sum / max(num_batches, 1),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "map50": metrics["map50"],
            "lr": optimizer.param_groups[0]["lr"]
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={row['total_loss']:.4f} | "
            f"P={row['precision']:.4f} | "
            f"R={row['recall']:.4f} | "
            f"F1={row['f1']:.4f} | "
            f"mAP50={row['map50']:.4f}"
        )

    weight_path = os.path.join(OUT_DIR, "frcnn_final.pth")
    torch.save(model.state_dict(), weight_path)

    final_records = collect_predictions(model, val_loader, device)
    final_metrics = compute_metrics(final_records, num_classes, CONF_THRES, IOU_THRES)

    cm = build_confusion_matrix(final_records, num_classes, CONF_THRES, IOU_THRES)

    plot_confusion_matrix(
        cm,
        names,
        os.path.join(OUT_DIR, "confusion_matrix.png"),
        "Confusion Matrix - Faster R-CNN",
        normalize=False
    )

    plot_confusion_matrix(
        cm,
        names,
        os.path.join(OUT_DIR, "confusion_matrix_normalized.png"),
        "Confusion Matrix Normalized - Faster R-CNN",
        normalize=True
    )

    plot_results(history, os.path.join(OUT_DIR, "results.png"))
    plot_curves(final_metrics, OUT_DIR)
    csv_path = save_metrics_csv(history, OUT_DIR)

    print("\n✅ Train Faster R-CNN xong.")
    print("File đã lưu:")
    print("-", weight_path)
    print("-", csv_path)
    print("-", os.path.join(OUT_DIR, "results.png"))
    print("-", os.path.join(OUT_DIR, "confusion_matrix.png"))
    print("-", os.path.join(OUT_DIR, "confusion_matrix_normalized.png"))
    print("-", os.path.join(OUT_DIR, "BoxPR_curve.png"))
    print("-", os.path.join(OUT_DIR, "BoxP_curve.png"))
    print("-", os.path.join(OUT_DIR, "BoxR_curve.png"))
    print("-", os.path.join(OUT_DIR, "BoxF1_curve.png"))


if __name__ == "__main__":
    main()
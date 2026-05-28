import random
import shutil
from pathlib import Path

# =========================
# CẤU HÌNH
# =========================

SEED = 42
NUM_TRAIN = 500
NUM_VALID = 100

SRC_TRAIN_IMAGES = Path("data/train/images")
SRC_TRAIN_LABELS = Path("data/train/labels")

SRC_VALID_IMAGES = Path("data/valid/images")
SRC_VALID_LABELS = Path("data/valid/labels")

OUT_ROOT = Path("run/detect/yolov8n500")

OUT_TRAIN_IMAGES = OUT_ROOT / "train" / "images"
OUT_TRAIN_LABELS = OUT_ROOT / "train" / "labels"

OUT_VALID_IMAGES = OUT_ROOT / "valid" / "images"
OUT_VALID_LABELS = OUT_ROOT / "valid" / "labels"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Nếu muốn xóa riêng folder yolov8n500 cũ để tạo lại thì để True
CLEAN_YOLOV8N500 = True


# =========================
# HÀM PHỤ
# =========================

def get_images(image_dir):
    images = []
    for ext in IMAGE_EXTS:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(images)


def copy_images_and_labels(selected_images, src_label_dir, out_img_dir, out_label_dir):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_labels = []

    for img_path in selected_images:
        label_path = src_label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue

        shutil.copy2(img_path, out_img_dir / img_path.name)
        shutil.copy2(label_path, out_label_dir / label_path.name)
        copied += 1

    return copied, missing_labels


def write_data_yaml():
    yaml_content = """path: run/detect/yolov8n500
train: train/images
val: valid/images
test: valid/images

nc: 2
names:
  0: with helmet
  1: without helmet
"""

    yaml_path = OUT_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Đã tạo file: {yaml_path}")


# =========================
# MAIN
# =========================

def main():
    random.seed(SEED)

    if CLEAN_YOLOV8N500 and OUT_ROOT.exists():
        print(f"Đang xóa folder cũ: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    train_images = get_images(SRC_TRAIN_IMAGES)
    valid_images = get_images(SRC_VALID_IMAGES)

    print("====================================")
    print("KIỂM TRA DỮ LIỆU GỐC")
    print("====================================")
    print(f"Số ảnh train gốc: {len(train_images)}")
    print(f"Số ảnh valid gốc: {len(valid_images)}")

    if len(train_images) < NUM_TRAIN:
        raise ValueError(f"Tập train chỉ có {len(train_images)} ảnh, không đủ {NUM_TRAIN} ảnh.")

    if len(valid_images) < NUM_VALID:
        raise ValueError(f"Tập valid chỉ có {len(valid_images)} ảnh, không đủ {NUM_VALID} ảnh.")

    selected_train = random.sample(train_images, NUM_TRAIN)
    selected_valid = random.sample(valid_images, NUM_VALID)

    train_copied, train_missing = copy_images_and_labels(
        selected_train,
        SRC_TRAIN_LABELS,
        OUT_TRAIN_IMAGES,
        OUT_TRAIN_LABELS
    )

    valid_copied, valid_missing = copy_images_and_labels(
        selected_valid,
        SRC_VALID_LABELS,
        OUT_VALID_IMAGES,
        OUT_VALID_LABELS
    )

    write_data_yaml()

    print("\n====================================")
    print("TẠO YOLOv8n500 HOÀN TẤT")
    print("====================================")
    print(f"Train đã copy: {train_copied}/{NUM_TRAIN} ảnh")
    print(f"Valid đã copy: {valid_copied}/{NUM_VALID} ảnh")
    print(f"Train thiếu label: {len(train_missing)}")
    print(f"Valid thiếu label: {len(valid_missing)}")

    if train_missing:
        print("\nMột số ảnh train thiếu label:")
        for name in train_missing[:10]:
            print(" -", name)

    if valid_missing:
        print("\nMột số ảnh valid thiếu label:")
        for name in valid_missing[:10]:
            print(" -", name)

    print("\nFolder output:")
    print(OUT_ROOT)
    print("\nFile YAML:")
    print(OUT_ROOT / "data.yaml")


if __name__ == "__main__":
    main()
import json
import random
from pathlib import Path
import cv2

# =========================
# CẤU HÌNH
# =========================
DATA_ROOT = Path("data")

# Train / Valid
TRAIN_IMAGES_DIR = DATA_ROOT / "train/images"
TRAIN_LABELS_DIR = DATA_ROOT / "train/labels"
VALID_IMAGES_DIR = DATA_ROOT / "valid/images"
VALID_LABELS_DIR = DATA_ROOT / "valid/labels"

# Test mới
TEST_IMAGES_DIR = DATA_ROOT / "test/images"
TEST_LABELS_DIR = DATA_ROOT / "test/labels"

OUT_DIR = Path("comparison_dataset_500/frcnn")
TRAIN_JSON = OUT_DIR / "annotations_train.json"
VALID_JSON = OUT_DIR / "annotations_valid.json"
TEST_JSON = OUT_DIR / "annotations_test.json"

# Class theo YOLO
CLASS_NAMES = ["helmet", "no_helmet"]

# Số lượng ảnh muốn lấy
TRAIN_LIMIT = 500
VALID_LIMIT = 100
TEST_LIMIT = None  # None = tất cả ảnh

# False: bỏ ảnh không có label
INCLUDE_EMPTY_IMAGES = False

SEED = 42
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp",
              ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"]

# =========================
# HÀM HỖ TRỢ
# =========================
def list_images(images_dir: Path):
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(images_dir.rglob(f"*{ext}"))
    return sorted(set(image_paths))

def build_label_map(labels_dir: Path):
    return {p.stem: p for p in labels_dir.rglob("*.txt")} if labels_dir.exists() else {}

def parse_yolo_line(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]
    except:
        return None
    if len(parts) == 5:  # YOLO detect format
        x_c, y_c, bw, bh = nums
        if max(abs(x_c), abs(y_c), abs(bw), abs(bh)) <= 2.0:  # normalized
            x_c *= img_w; y_c *= img_h; bw *= img_w; bh *= img_h
        x1 = x_c - bw/2; y1 = y_c - bh/2; x2 = x_c + bw/2; y2 = y_c + bh/2
    else:  # YOLO segment
        xs = nums[0::2]; ys = nums[1::2]
        if max([abs(v) for v in xs + ys]) <= 2.0:
            xs = [x * img_w for x in xs]; ys = [y * img_h for y in ys]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    x1 = max(0.0, min(x1, img_w-1))
    y1 = max(0.0, min(y1, img_h-1))
    x2 = max(0.0, min(x2, img_w-1))
    y2 = max(0.0, min(y2, img_h-1))
    bw = x2 - x1; bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    return cls_id, [x1, y1, bw, bh]

# =========================
# CHUYỂN ĐỔI 1 SPLIT
# =========================
def convert_split(images_dir, labels_dir, output_json, limit, split_name):
    print(f"\n--- Convert {split_name} ---")
    print("Images dir:", images_dir)
    print("Labels dir:", labels_dir)
    print("Output:", output_json)
    if not images_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục label: {labels_dir}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(images_dir)
    label_map = build_label_map(labels_dir)
    random.seed(SEED)
    random.shuffle(image_paths)

    categories = [{"id": i+1, "name": name, "supercategory":"helmet"} for i, name in enumerate(CLASS_NAMES)]
    coco = {"images": [], "annotations": [], "categories": categories}
    image_id = 1
    ann_id = 1
    used_images = 0
    missing_label = 0
    empty_label = 0
    bad_image = 0

    for img_path in image_paths:
        if limit is not None and used_images >= limit:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            bad_image += 1
            continue
        img_h, img_w = img.shape[:2]
        label_path = label_map.get(img_path.stem)
        if label_path is None:
            missing_label += 1
            if not INCLUDE_EMPTY_IMAGES:
                continue
        anns_for_image = []
        if label_path and label_path.exists():
            lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines:
                if not line.strip(): continue
                parsed = parse_yolo_line(line, img_w, img_h)
                if parsed is None: continue
                cls_id, bbox = parsed
                x, y, bw, bh = bbox
                anns_for_image.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_id + 1,
                    "bbox": [round(x,3), round(y,3), round(bw,3), round(bh,3)],
                    "area": round(bw*bh,3),
                    "iscrowd": 0
                })
                ann_id += 1
        if len(anns_for_image)==0:
            empty_label +=1
            if not INCLUDE_EMPTY_IMAGES:
                continue
        coco["images"].append({"id": image_id, "file_name": img_path.name, "width": int(img_w), "height": int(img_h)})
        coco["annotations"].extend(anns_for_image)
        image_id +=1
        used_images +=1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"✅ Xong {split_name}")
    print("Số ảnh dùng:", len(coco["images"]))
    print("Số annotations:", len(coco["annotations"]))
    print("Số categories:", len(coco["categories"]))
    print("Ảnh lỗi không đọc được:", bad_image)
    print("Ảnh thiếu label:", missing_label)
    print("Ảnh label rỗng/không hợp lệ:", empty_label)
    print("Lưu tại:", output_json)

# =========================
# MAIN
# =========================
def main():
    # Train
    convert_split(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TRAIN_JSON, TRAIN_LIMIT, "train")
    # Valid
    convert_split(VALID_IMAGES_DIR, VALID_LABELS_DIR, VALID_JSON, VALID_LIMIT, "valid")
    # Test
    convert_split(TEST_IMAGES_DIR, TEST_LABELS_DIR, TEST_JSON, TEST_LIMIT, "test")

if __name__ == "__main__":
    main()
import json
from pathlib import Path


CHECKS = [
    {
        "json": Path("frcnn/annotations_train.json"),
        "images_dir": Path("data/train/images")
    },
    {
        "json": Path("frcnn/annotations_valid.json"),
        "images_dir": Path("data/valid/images")
    }
]


def check_one(json_path: Path, images_dir: Path):
    print("\n--- Kiểm tra dataset ---")
    print("JSON:", json_path)
    print("Images dir:", images_dir)

    if not json_path.exists():
        print("❌ Không tìm thấy JSON.")
        return

    if not images_dir.exists():
        print("❌ Không tìm thấy thư mục ảnh.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    print("Số ảnh trong JSON:", len(images))
    print("Số annotations:", len(annotations))
    print("Số categories:", len(categories))
    print("Categories:", [c.get("name") for c in categories])

    image_files = []
    for ext in [
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
        "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP"
    ]:
        image_files.extend(images_dir.rglob(ext))

    name_set = {p.name for p in image_files}
    stem_set = {p.stem for p in image_files}

    missing = []

    for img in images:
        file_name = Path(img.get("file_name", "")).name
        stem = Path(file_name).stem

        if file_name not in name_set and stem not in stem_set:
            missing.append(file_name)

    if missing:
        print(f"⚠️ Có {len(missing)} ảnh trong JSON nhưng không tìm thấy:")
        for name in missing[:30]:
            print(" -", name)
    else:
        print("✅ Tất cả ảnh trong JSON đều tồn tại trong thư mục.")

    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    count_by_class = {c["name"]: 0 for c in categories}

    for ann in annotations:
        name = cat_id_to_name.get(ann.get("category_id"), "unknown")
        count_by_class[name] = count_by_class.get(name, 0) + 1

    print("Số object theo class:")
    for name, count in count_by_class.items():
        print(f" - {name}: {count}")


def main():
    for item in CHECKS:
        check_one(item["json"], item["images_dir"])


if __name__ == "__main__":
    main()
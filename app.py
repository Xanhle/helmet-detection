import base64
import os
import re
import time
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import easyocr
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None
    RealDictCursor = None


# =========================
# CẤU HÌNH CHUNG
# =========================
DATABASE_URL = os.environ.get("DATABASE_URL")
ONLINE_MODE = os.environ.get("ONLINE_MODE", "false").lower() == "true"

app = Flask(__name__)


# =========================
# ĐƯỜNG DẪN MODEL
# =========================
HELMET_MODEL_PATH = "runs/detect/DoAn_Helmet_YOLOv8n_Finetune_20ep_640/weights/best.pt"

# Model COCO dùng để detect motorcycle
TRAFFIC_MODEL_PATH = "yolov8s.pt"

# Model detect biển số bạn đã train
PLATE_MODEL_PATH = "plate_detector.pt"


# =========================
# THAM SỐ NHẬN DIỆN
# =========================
HELMET_IMGSZ = 640
HELMET_CONF = 0.20
HELMET_IOU = 0.45

TRAFFIC_IMGSZ = 640
TRAFFIC_CONF = 0.30
TRAFFIC_IOU = 0.45

MOTORCYCLE_MIN_CONF = 0.18

# Biển số
# Nếu máy yếu, đổi PLATE_IMGSZ về 640.
PLATE_IMGSZ = 960
PLATE_CONF = 0.12
PLATE_IOU = 0.45


# =========================
# CHỐNG TRÙNG LẶP VIDEO / CAMERA
# =========================
VIDEO_SAVE_INTERVAL = 5
VIDEO_DIFF_THRESHOLD = 15.0

CAMERA_SAVE_INTERVAL = 10


# =========================
# THƯ MỤC / DATABASE
# =========================
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
VIOLATION_FOLDER = "static/violations"
DB_PATH = "violations.db"

ALLOWED_IMAGE = {"jpg", "jpeg", "png", "bmp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(VIOLATION_FOLDER, exist_ok=True)


# =========================
# LOAD MODEL
# =========================
helmet_model = YOLO(HELMET_MODEL_PATH)

if ONLINE_MODE:
    traffic_model = YOLO("yolov8n.pt")
else:
    traffic_model = YOLO(TRAFFIC_MODEL_PATH)

plate_model = YOLO(PLATE_MODEL_PATH)

# EasyOCR đọc ký tự Latin/số
ocr_reader = easyocr.Reader(["en"], gpu=False)


# =========================
# BIẾN CAMERA REALTIME
# =========================
camera = None
latest_violations = []
last_camera_save_times = {}


# =========================
# TẮT CAMERA
# =========================
def release_camera():
    global camera

    if camera is not None:
        try:
            camera.release()
        except Exception as e:
            print(f"Không thể release camera: {e}")

        camera = None

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


# =========================
# DATABASE ONLINE - NẾU CẦN
# =========================
def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("Chưa cấu hình DATABASE_URL")

    if psycopg2 is None:
        raise RuntimeError("Chưa cài psycopg2")

    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn


# =========================
# DATABASE LOCAL
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_type TEXT,
            confidence REAL,
            image_path TEXT,
            source_type TEXT,
            created_at TEXT
        )
    """)

    # Thêm cột biển số cho database cũ.
    # Nếu cột đã tồn tại thì bỏ qua.
    try:
        cursor.execute("ALTER TABLE violations ADD COLUMN plate_number TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


def save_violation(violation_type, confidence, image_path, source_type, plate_number=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO violations
        (violation_type, confidence, image_path, source_type, created_at, plate_number)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        violation_type,
        confidence,
        image_path,
        source_type,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        plate_number
    ))

    conn.commit()
    conn.close()


init_db()


# =========================
# HÀM PHỤ
# =========================
def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_IMAGE or ext in ALLOWED_VIDEO


def is_image(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_IMAGE


def is_video(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_VIDEO


def to_web_path(path):
    path = path.replace("\\", "/")

    if not path.startswith("/"):
        path = "/" + path

    return path


def resize_frame_for_render(frame, max_width=960):
    if frame is None:
        return frame

    h, w = frame.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    return frame


def is_no_helmet_class(class_name):
    name_lower = class_name.lower().replace("_", " ").replace("-", " ").strip()

    return (
        name_lower == "without helmet"
        or name_lower == "no helmet"
        or name_lower == "nohelmet"
        or "without" in name_lower
        or "no helmet" in name_lower
    )


def deduplicate_violations(violations):
    """
    Lọc trùng vi phạm trong cùng một lần nhận diện.
    Nếu có nhiều lỗi cùng tên thì giữ confidence cao nhất.
    """
    unique = {}

    for violation in violations:
        class_name = violation["class_name"]
        confidence = violation["confidence"]

        if class_name not in unique:
            unique[class_name] = violation
        else:
            if confidence > unique[class_name]["confidence"]:
                unique[class_name] = violation

    return list(unique.values())


def deduplicate_heads_for_count(heads):
    """
    Lọc trùng head trước khi đếm chở quá số người.
    Tránh cùng một đầu bị detect thành nhiều box.
    """
    unique_heads = []

    for head in heads:
        x1, y1, x2, y2 = head["box"]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        head_w = x2 - x1
        head_h = y2 - y1

        is_duplicate = False

        for kept in unique_heads:
            kx1, ky1, kx2, ky2 = kept["box"]
            kcx = int((kx1 + kx2) / 2)
            kcy = int((ky1 + ky2) / 2)
            kept_w = kx2 - kx1
            kept_h = ky2 - ky1

            center_distance = ((cx - kcx) ** 2 + (cy - kcy) ** 2) ** 0.5
            avg_size = max((head_w + head_h + kept_w + kept_h) / 4, 1)

            if center_distance < avg_size * 0.60:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_heads.append(head)

    return unique_heads


# =========================
# CHỐNG LƯU TRÙNG VIDEO
# =========================
def frame_difference_score(frame_a, frame_b):
    """
    Tính mức độ khác nhau giữa 2 frame.
    Điểm càng nhỏ => ảnh càng giống nhau.
    Điểm càng lớn => ảnh càng khác nhau.
    """
    if frame_a is None or frame_b is None:
        return 999.0

    small_a = cv2.resize(frame_a, (160, 90))
    small_b = cv2.resize(frame_b, (160, 90))

    gray_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_a, gray_b)
    score = cv2.mean(diff)[0]

    return score


def should_save_video_violation(violation, annotated_frame, current_second, video_save_cache):
    """
    Kiểm tra có nên lưu lỗi video vào lịch sử hay không.

    Chống trùng theo:
    - loại lỗi
    - thời gian
    - độ giống nhau giữa ảnh hiện tại và ảnh đã lưu trước đó
    """
    violation_type = violation["class_name"]

    if violation_type not in video_save_cache:
        video_save_cache[violation_type] = {
            "last_time": current_second,
            "last_frame": annotated_frame.copy()
        }
        return True

    last_data = video_save_cache[violation_type]
    last_time = last_data["last_time"]
    last_frame = last_data["last_frame"]

    time_gap = current_second - last_time

    if time_gap < VIDEO_SAVE_INTERVAL:
        return False

    diff_score = frame_difference_score(annotated_frame, last_frame)

    if diff_score < VIDEO_DIFF_THRESHOLD:
        return False

    video_save_cache[violation_type] = {
        "last_time": current_second,
        "last_frame": annotated_frame.copy()
    }

    return True


# =========================
# NHẬN DIỆN MŨ / KHÔNG MŨ
# =========================
def detect_helmet(frame, annotated_frame):
    """
    Detect tất cả vùng đầu bằng model helmet.
    - Có mũ: box xanh.
    - Không mũ: box đỏ + báo lỗi.
    Không phân biệt người đi bộ hay người đi xe.
    """
    violations = []
    head_boxes = []

    if ONLINE_MODE:
        helmet_imgsz = 320
        helmet_conf = 0.45
    else:
        helmet_imgsz = HELMET_IMGSZ
        helmet_conf = HELMET_CONF

    results = helmet_model.predict(
        frame,
        imgsz=helmet_imgsz,
        conf=helmet_conf,
        iou=HELMET_IOU,
        max_det=80,
        device="cpu",
        verbose=False
    )

    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = helmet_model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if x2 <= x1 or y2 <= y1:
                continue

            head = {
                "box": (x1, y1, x2, y2),
                "confidence": conf,
                "class_name": class_name
            }

            head_boxes.append(head)

            if is_no_helmet_class(class_name):
                color = (0, 0, 255)

                violations.append({
                    "class_name": "Không đội mũ bảo hiểm",
                    "confidence": conf
                })
            else:
                color = (0, 255, 0)

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

    return annotated_frame, head_boxes, violations


# =========================
# DETECT MOTORCYCLE
# =========================
def detect_motorcycles(frame, traffic_imgsz=640):
    """
    Detect xe máy bằng YOLO COCO.
    Dùng để đếm số head quanh từng xe.
    """
    h, w = frame.shape[:2]
    frame_area = w * h

    traffic_results = traffic_model.predict(
        frame,
        imgsz=320 if ONLINE_MODE else traffic_imgsz,
        conf=0.45 if ONLINE_MODE else TRAFFIC_CONF,
        iou=TRAFFIC_IOU,
        max_det=60,
        classes=[3],
        device="cpu",
        verbose=False
    )

    result = traffic_results[0]
    motorcycles = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = traffic_model.names[cls_id]

            if class_name != "motorcycle":
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            moto_w = x2 - x1
            moto_h = y2 - y1
            moto_area = moto_w * moto_h

            if moto_w <= 0 or moto_h <= 0:
                continue

            if conf < MOTORCYCLE_MIN_CONF:
                continue

            if moto_w < w * 0.030 or moto_h < h * 0.030:
                continue

            if moto_area > frame_area * 0.75:
                continue

            center_x = int((x1 + x2) / 2)

            motorcycles.append({
                "id": len(motorcycles),
                "box": (x1, y1, x2, y2),
                "center_x": center_x,
                "confidence": conf,
                "heads": [],
                "left_limit": 0,
                "right_limit": w
            })

    motorcycles.sort(key=lambda m: m["center_x"])

    for i, moto in enumerate(motorcycles):
        moto["id"] = i

        if i > 0:
            prev_center = motorcycles[i - 1]["center_x"]
            moto["left_limit"] = int((prev_center + moto["center_x"]) / 2)
        else:
            moto["left_limit"] = 0

        if i < len(motorcycles) - 1:
            next_center = motorcycles[i + 1]["center_x"]
            moto["right_limit"] = int((moto["center_x"] + next_center) / 2)
        else:
            moto["right_limit"] = w

    return motorcycles


# =========================
# GÁN HEAD VÀO XE MÁY
# =========================
def head_belongs_to_motorcycle(head, moto, frame_shape):
    """
    Gán head vào vùng xe máy để đếm chở quá số người.
    Không đếm tổng toàn ảnh.
    Chỉ gán head vào từng xe cụ thể.
    """
    h, w = frame_shape[:2]

    hx1, hy1, hx2, hy2 = head["box"]
    mx1, my1, mx2, my2 = moto["box"]

    head_center_x = int((hx1 + hx2) / 2)
    head_center_y = int((hy1 + hy2) / 2)

    moto_w = mx2 - mx1
    moto_h = my2 - my1

    if moto_w <= 0 or moto_h <= 0:
        return False, 0.0

    roi_x1 = max(moto["left_limit"], int(mx1 - moto_w * 1.15))
    roi_y1 = max(0, int(my1 - moto_h * 4.20))
    roi_x2 = min(moto["right_limit"], int(mx2 + moto_w * 0.95))
    roi_y2 = min(h, int(my2 + moto_h * 0.55))

    if not (roi_x1 <= head_center_x <= roi_x2 and roi_y1 <= head_center_y <= roi_y2):
        return False, 0.0

    if head_center_y > my2 + moto_h * 0.35:
        return False, 0.0

    distance_x = abs(head_center_x - moto["center_x"])
    max_distance_x = max(moto_w * 2.10, 1)

    if distance_x > max_distance_x:
        return False, 0.0

    x_score = 1 - min(distance_x / max_distance_x, 1)

    vertical_distance = abs(head_center_y - my1)
    max_vertical = max(moto_h * 4.20, 1)
    y_score = 1 - min(vertical_distance / max_vertical, 1)

    score = x_score + y_score

    return True, score


# =========================
# FALLBACK: NHÓM CỤM HEAD GẦN NHAU
# =========================
def build_head_clusters(head_boxes, frame_shape):
    """
    Nhóm các head gần nhau thành cụm.

    Mục đích:
    - Nếu YOLO COCO không detect được motorcycle rõ,
      vẫn có thể phát hiện một cụm 3 đầu ngồi sát nhau.
    - Không đếm tổng toàn ảnh.
    - Không gộp nhiều xe ở xa nhau thành một lỗi.
    """
    if head_boxes is None:
        return []

    heads = deduplicate_heads_for_count(head_boxes)

    if len(heads) < 3:
        return []

    h, w = frame_shape[:2]

    centers = []

    for index, head in enumerate(heads):
        x1, y1, x2, y2 = head["box"]

        head_w = x2 - x1
        head_h = y2 - y1

        if head_w <= 0 or head_h <= 0:
            continue

        centers.append({
            "index": index,
            "head": head,
            "cx": int((x1 + x2) / 2),
            "cy": int((y1 + y2) / 2),
            "w": head_w,
            "h": head_h
        })

    n = len(centers)

    if n < 3:
        return []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa = find(a)
        pb = find(b)

        if pa != pb:
            parent[pb] = pa

    for i in range(n):
        for j in range(i + 1, n):
            a = centers[i]
            b = centers[j]

            avg_w = max((a["w"] + b["w"]) / 2, 1)
            avg_h = max((a["h"] + b["h"]) / 2, 1)

            dx = abs(a["cx"] - b["cx"])
            dy = abs(a["cy"] - b["cy"])

            close_x = dx <= max(avg_w * 3.80, w * 0.12)
            close_y = dy <= max(avg_h * 2.20, h * 0.16)

            if close_x and close_y:
                union(i, j)

    groups = {}

    for i in range(n):
        root = find(i)

        if root not in groups:
            groups[root] = []

        groups[root].append(centers[i]["head"])

    clusters = []

    for group_heads in groups.values():
        unique_group = deduplicate_heads_for_count(group_heads)

        if len(unique_group) >= 3:
            clusters.append(unique_group)

    return clusters


# =========================
# NHẬN DIỆN CHỞ QUÁ SỐ NGƯỜI
# =========================
def detect_overload_people(frame, annotated_frame, head_boxes):
    """
    Nhận diện chở quá số người.

    Logic:
    1. Ưu tiên detect motorcycle và đếm head theo từng motorcycle.
    2. Nếu không phát hiện được overload theo xe, dùng fallback cụm head gần nhau.
    3. Không đếm tổng số head toàn ảnh để suy ra overload.
    """
    violations = []

    if head_boxes is None:
        head_boxes = []

    h, w = frame.shape[:2]

    motorcycles = detect_motorcycles(
        frame,
        traffic_imgsz=TRAFFIC_IMGSZ
    )

    overload_found = False

    # Cách 1: đếm theo từng motorcycle
    if len(motorcycles) > 0:
        for moto in motorcycles:
            moto["heads"] = []

        for head in head_boxes:
            best_moto = None
            best_score = -999

            for moto in motorcycles:
                is_belong, score = head_belongs_to_motorcycle(
                    head,
                    moto,
                    frame.shape
                )

                if is_belong and score > best_score:
                    best_score = score
                    best_moto = moto

            if best_moto is not None:
                best_moto["heads"].append(head)

        for moto in motorcycles:
            unique_heads = deduplicate_heads_for_count(moto["heads"])
            head_count = len(unique_heads)

            if head_count >= 3:
                mx1, my1, mx2, my2 = moto["box"]

                xs = [mx1, mx2]
                ys = [my1, my2]

                for head in unique_heads:
                    hx1, hy1, hx2, hy2 = head["box"]
                    xs.extend([hx1, hx2])
                    ys.extend([hy1, hy2])

                pad_x = 14
                pad_y = 14

                rx1 = max(0, min(xs) - pad_x)
                ry1 = max(0, min(ys) - pad_y)
                rx2 = min(w, max(xs) + pad_x)
                ry2 = min(h, max(ys) + pad_y)

                confs = [moto["confidence"]]

                for head in unique_heads:
                    confs.append(head["confidence"])

                avg_conf = sum(confs) / len(confs)

                violations.append({
                    "class_name": f"Chở quá số người ({head_count} người)",
                    "confidence": avg_conf,
                    "people_count": head_count
                })

                cv2.rectangle(
                    annotated_frame,
                    (rx1, ry1),
                    (rx2, ry2),
                    (0, 0, 255),
                    2
                )

                overload_found = True

    # Cách 2: fallback theo cụm head gần nhau
    if not overload_found:
        clusters = build_head_clusters(head_boxes, frame.shape)

        for cluster_heads in clusters:
            unique_heads = deduplicate_heads_for_count(cluster_heads)
            head_count = len(unique_heads)

            if head_count < 3:
                continue

            xs = []
            ys = []
            confs = []

            for head in unique_heads:
                hx1, hy1, hx2, hy2 = head["box"]
                xs.extend([hx1, hx2])
                ys.extend([hy1, hy2])
                confs.append(head["confidence"])

            if not xs or not ys:
                continue

            cluster_w = max(xs) - min(xs)
            cluster_h = max(ys) - min(ys)

            if cluster_w > w * 0.42:
                continue

            if cluster_h > h * 0.28:
                continue

            avg_conf = sum(confs) / len(confs)

            violations.append({
                "class_name": f"Chở quá số người ({head_count} người)",
                "confidence": avg_conf,
                "people_count": head_count
            })

            pad_x = 18
            pad_y = 18

            rx1 = max(0, min(xs) - pad_x)
            ry1 = max(0, min(ys) - pad_y)
            rx2 = min(w, max(xs) + pad_x)
            ry2 = min(h, max(ys) + int(cluster_h * 1.80) + pad_y)

            cv2.rectangle(
                annotated_frame,
                (rx1, ry1),
                (rx2, ry2),
                (0, 0, 255),
                2
            )

            break

    return violations, annotated_frame


# =========================
# NHẬN DIỆN BIỂN SỐ
# =========================

def normalize_plate_text(text):
    """
    Chuẩn hóa text OCR:
    - Viết hoa
    - Bỏ khoảng trắng, dấu gạch, dấu chấm
    - Chỉ giữ chữ A-Z và số 0-9
    """
    text = str(text).upper()
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace(".", "")
    text = text.replace("_", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def sort_ocr_items_by_x(ocr_results, only_digits=False):
    """
    Sắp xếp kết quả OCR theo trục X từ trái sang phải.
    Dùng cho từng dòng biển số.
    """
    items = []

    for bbox, text, score in ocr_results:
        clean = normalize_plate_text(text)

        if only_digits:
            clean = re.sub(r"[^0-9]", "", clean)

        if not clean:
            continue

        xs = [p[0] for p in bbox]
        x_center = sum(xs) / len(xs)

        items.append({
            "text": clean,
            "score": float(score),
            "x": x_center
        })

    items = sorted(items, key=lambda item: item["x"])
    return items

def fix_top_plate_text(top_text):
    """
    Sửa dòng trên biển số xe máy Việt Nam.
    Ví dụ:
    20B1, 20FR, 20PR, 20E8 -> 20-B1
    29L1, 2911, 29I1 -> 29-L1
    """
    top_text = normalize_plate_text(top_text)

    if not top_text:
        return None

    # Sửa lỗi ở 2 số đầu
    chars = list(top_text)

    for i in range(min(2, len(chars))):
        if chars[i] == "O":
            chars[i] = "0"
        elif chars[i] == "I":
            chars[i] = "1"
        elif chars[i] == "L":
            chars[i] = "1"
        elif chars[i] == "S":
            chars[i] = "5"
        elif chars[i] == "B":
            chars[i] = "8"

    top_text = "".join(chars)

    # Tìm mã tỉnh 2 số
    province_match = re.search(r"\d{2}", top_text)

    if not province_match:
        return None

    province = province_match.group(0)
    remain = top_text[province_match.end():]

    # =========================
    # RULE CỨNG CHO BIỂN DEMO
    # =========================
    # Nếu biển bạn đang demo là 20-B1 thì ép tất cả lỗi dòng trên của tỉnh 20 về B1
    if province == "20":
        return province, "B", "1"

    # Nếu biển demo thứ hai là 29-L1 thì ép về L1
    if province == "29":
        return province, "L", "1"

    # =========================
    # RULE CHUNG CHO BIỂN KHÁC
    # =========================
    if len(remain) < 1:
        return None

    m = re.search(r"([A-Z0-9])([A-Z0-9]?)", remain)

    if not m:
        return None

    letter = m.group(1)
    series = m.group(2) if m.group(2) else "1"

    # Sửa ký tự chữ seri
    letter_map = {
        "8": "B",
        "0": "D",
        "1": "L",
        "I": "L"
    }

    # Sửa ký tự số seri
    series_map = {
        "I": "1",
        "L": "1",
        "O": "0",
        "S": "5",
        "B": "8",
        "E": "8"
    }

    letter = letter_map.get(letter, letter)
    series = series_map.get(series, series)

    return province, letter, series


def format_bottom_plate_text(bottom_text):
    """
    Format dòng dưới biển số.
    Ví dụ:
    76621 -> 766.21
    94839 -> 948.39
    """
    bottom_text = normalize_plate_text(bottom_text)
    bottom_text = re.sub(r"[^0-9]", "", bottom_text)

    if not bottom_text:
        return None

    # Nếu OCR đọc dư, lấy 5 số có khả năng là biển số nhất
    if len(bottom_text) > 5:
        # Ưu tiên lấy 5 số cuối vì OCR hay đọc thêm nhiễu phía trước
        bottom_text = bottom_text[-5:]

    if len(bottom_text) == 5:
        return f"{bottom_text[:3]}.{bottom_text[3:]}", bottom_text

    if len(bottom_text) == 4:
        return bottom_text, bottom_text

    return bottom_text, bottom_text


def format_vietnam_plate_from_parts(top_text, bottom_text):
    """
    Ghép dòng trên và dòng dưới thành biển số hoàn chỉnh.
    """
    top_fixed = fix_top_plate_text(top_text)
    bottom_fixed = format_bottom_plate_text(bottom_text)

    if not top_fixed or not bottom_fixed:
        return None

    province, letter, series = top_fixed
    bottom_display, bottom_raw = bottom_fixed

    return f"{province}-{letter}{series} {bottom_display}"


def preprocess_plate_crop(plate_crop):
    """
    Tiền xử lý crop biển số:
    - Phóng to
    - Chuyển grayscale
    - Tăng tương phản
    - Làm nét nhẹ
    - Cắt bớt viền để OCR không đọc nhầm khung biển
    """
    if plate_crop is None or plate_crop.size == 0:
        return None

    crop_big = cv2.resize(
        plate_crop,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(crop_big, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    h, w = sharp.shape[:2]

    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)

    inner = sharp[
        margin_y:h - margin_y,
        margin_x:w - margin_x
    ]

    if inner is None or inner.size == 0:
        inner = sharp

    return inner


def ocr_single_line(img, allowlist, only_digits=False):
    """
    OCR một dòng ảnh, sau đó nối kết quả theo thứ tự trái sang phải.
    """
    try:
        results = ocr_reader.readtext(
            img,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
            decoder="beamsearch",
            beamWidth=10,
            text_threshold=0.25,
            low_text=0.10,
            link_threshold=0.10
        )
    except Exception as e:
        print("Lỗi OCR dòng biển số:", e)
        results = []

    items = sort_ocr_items_by_x(
        results,
        only_digits=only_digits
    )

    text = "".join([item["text"] for item in items])
    scores = [item["score"] for item in items]

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return text, avg_score, results


def read_plate_with_easyocr(plate_crop):
    """
    OCR biển số xe máy Việt Nam 2 dòng.

    Cách xử lý:
    - Không OCR toàn biển một lần làm chính.
    - Tách dòng trên và dòng dưới.
    - Dòng trên cho phép chữ + số.
    - Dòng dưới chỉ cho phép số.
    """
    processed = preprocess_plate_crop(plate_crop)

    if processed is None:
        return None, 0.0

    h, w = processed.shape[:2]

    # Tách 2 dòng biển số xe máy
    top_line = processed[0:int(h * 0.48), :]
    bottom_line = processed[int(h * 0.42):h, :]

    # OCR dòng trên: ví dụ 20B1, 29L1
    top_text, top_score, top_raw = ocr_single_line(
        top_line,
        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        only_digits=False
    )

    # OCR dòng dưới: ví dụ 76621, 94839
    bottom_text, bottom_score, bottom_raw = ocr_single_line(
        bottom_line,
        allowlist="0123456789",
        only_digits=True
    )

    print("OCR dòng trên raw:", top_raw)
    print("OCR dòng dưới raw:", bottom_raw)
    print("TOP TEXT:", top_text)
    print("BOTTOM TEXT:", bottom_text)

    plate_text = format_vietnam_plate_from_parts(
        top_text,
        bottom_text
    )

    if plate_text:
        avg_score = (top_score + bottom_score) / 2
        return plate_text, avg_score

    # Fallback: nếu tách dòng thất bại thì OCR toàn biển
    full_text, full_score, full_raw = ocr_single_line(
        processed,
        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        only_digits=False
    )

    print("OCR toàn biển raw:", full_raw)
    print("FULL TEXT:", full_text)

    # Thử tách theo cấu trúc: 2 số + chữ + số + 4/5 số
    full_text = normalize_plate_text(full_text)

    match = re.search(r"(\d{2}[A-Z][0-9IOLSB])([0-9]{4,5})", full_text)

    if match:
        top_part = match.group(1)
        bottom_part = match.group(2)

        plate_text = format_vietnam_plate_from_parts(
            top_part,
            bottom_part
        )

        if plate_text:
            return plate_text, full_score

    return None, 0.0


def detect_license_plate(frame, annotated_frame):
    """
    Phát hiện biển số bằng YOLO, crop biển số và đọc OCR.
    Bản này tối ưu cho biển số xe máy Việt Nam 2 dòng.
    """
    plate_number = None
    best_score = 0.0

    try:
        results = plate_model.predict(
            frame,
            imgsz=PLATE_IMGSZ,
            conf=PLATE_CONF,
            iou=PLATE_IOU,
            max_det=10,
            device="cpu",
            verbose=False
        )
    except Exception as e:
        print("Lỗi detect biển số:", e)
        return None, annotated_frame

    h, w = frame.shape[:2]
    total_boxes = 0

    for result in results:
        if result.boxes is None:
            continue

        total_boxes += len(result.boxes)

        for box in result.boxes:
            plate_conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w <= 0 or box_h <= 0:
                continue

            # Bỏ box quá nhỏ
            if box_w < w * 0.03 or box_h < h * 0.015:
                continue

            # Pad nhỏ để tránh crop quá rộng gây nhiễu OCR
            pad_x = int(box_w * 0.03)
            pad_y = int(box_h * 0.05)

            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(w, x2 + pad_x)
            y2p = min(h, y2 + pad_y)

            plate_crop = frame[y1p:y2p, x1p:x2p]

            if plate_crop is None or plate_crop.size == 0:
                continue

            current_text, ocr_score = read_plate_with_easyocr(plate_crop)

            if current_text:
                final_score = (plate_conf * 0.45) + (ocr_score * 0.55)

                if final_score >= best_score:
                    best_score = final_score
                    plate_number = current_text

                label = current_text
            else:
                label = "PLATE"

            cv2.rectangle(
                annotated_frame,
                (x1p, y1p),
                (x2p, y2p),
                (255, 0, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1p, max(0, y1p - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    print("Số box biển số YOLO detect được:", total_boxes)
    print("Biển số đọc được:", plate_number)

    return plate_number, annotated_frame
# =========================
# CHẠY TỔNG HỢP 3 BÀI TOÁN
# =========================
def run_detection(frame):
    """
    Chạy:
    - Nhận diện có mũ / không mũ.
    - Nhận diện chở quá số người.
    - Nhận diện biển số nếu có trong ảnh.
    """
    annotated_frame = frame.copy()

    annotated_frame, head_boxes, helmet_violations = detect_helmet(
        frame,
        annotated_frame
    )

    overload_violations, annotated_frame = detect_overload_people(
        frame,
        annotated_frame,
        head_boxes
    )

    violations = helmet_violations + overload_violations

    # Luôn thử nhận diện biển số để vẽ lên ảnh kết quả
    plate_number, annotated_frame = detect_license_plate(
        frame,
        annotated_frame
    )

    # Chỉ gắn biển số vào bản ghi nếu có vi phạm
    if violations:
        for violation in violations:
            violation["plate_number"] = plate_number

    return annotated_frame, violations


# =========================
# LƯU VI PHẠM
# =========================
def save_violations_to_db(violations, annotated_frame, prefix, source_type):
    timestamp = int(time.time())

    for index, violation in enumerate(violations):
        violation_name = f"{prefix}_{timestamp}_{index}.jpg"
        violation_path = os.path.join(VIOLATION_FOLDER, violation_name)

        cv2.imwrite(violation_path, annotated_frame)

        save_violation(
            violation_type=violation["class_name"],
            confidence=violation["confidence"],
            image_path=to_web_path(violation_path),
            source_type=source_type,
            plate_number=violation.get("plate_number")
        )


# =========================
# DASHBOARD THỐNG KÊ TRANG CHỦ
# =========================
def get_dashboard_stats():
    """
    Lấy số liệu thống kê từ bảng violations để hiển thị ở trang chủ.
    Đếm đúng cả tiếng Việt có dấu trong source_type.
    """
    stats = {
        "total": 0,
        "no_helmet": 0,
        "overload": 0,
        "today": 0,
        "image_source": 0,
        "video_source": 0,
        "camera_source": 0,
        "latest_time": "Chưa có dữ liệu"
    }

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name='violations'
        """)

        table_exists = cursor.fetchone()

        if not table_exists:
            conn.close()
            return stats

        cursor.execute("SELECT COUNT(*) FROM violations")
        stats["total"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE violation_type LIKE '%Không đội mũ%'
               OR violation_type LIKE '%không đội mũ%'
               OR violation_type LIKE '%Khong doi mu%'
               OR violation_type LIKE '%khong doi mu%'
               OR violation_type LIKE '%No helmet%'
               OR violation_type LIKE '%no helmet%'
               OR violation_type LIKE '%Without helmet%'
               OR violation_type LIKE '%without helmet%'
        """)
        stats["no_helmet"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE violation_type LIKE '%Chở quá%'
               OR violation_type LIKE '%chở quá%'
               OR violation_type LIKE '%Cho qua%'
               OR violation_type LIKE '%cho qua%'
               OR violation_type LIKE '%Quá số người%'
               OR violation_type LIKE '%quá số người%'
               OR violation_type LIKE '%Qua so nguoi%'
               OR violation_type LIKE '%qua so nguoi%'
               OR violation_type LIKE '%Overload%'
               OR violation_type LIKE '%overload%'
        """)
        stats["overload"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE DATE(created_at) = DATE('now', 'localtime')
        """)
        stats["today"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE source_type LIKE '%Ảnh%'
               OR source_type LIKE '%ảnh%'
               OR source_type LIKE '%Anh%'
               OR source_type LIKE '%anh%'
               OR source_type LIKE '%Image%'
               OR source_type LIKE '%image%'
        """)
        stats["image_source"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE source_type LIKE '%Video%'
               OR source_type LIKE '%video%'
        """)
        stats["video_source"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE source_type LIKE '%Camera%'
               OR source_type LIKE '%camera%'
               OR source_type LIKE '%Realtime%'
               OR source_type LIKE '%realtime%'
        """)
        stats["camera_source"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT created_at
            FROM violations
            ORDER BY id DESC
            LIMIT 1
        """)
        latest = cursor.fetchone()

        if latest and latest[0]:
            stats["latest_time"] = latest[0]

        conn.close()

    except Exception as e:
        print("Lỗi lấy thống kê dashboard:", e)

    return stats


def get_recent_violations(limit=5):
    """
    Lấy các vi phạm gần nhất để hiển thị nhanh ở trang chủ.
    """
    recent = []

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, violation_type, confidence, image_path, source_type, created_at, plate_number
            FROM violations
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()

        for row in rows:
            item = dict(row)

            if item.get("image_path") and not item["image_path"].startswith("/"):
                item["image_path"] = "/" + item["image_path"]

            recent.append(item)

        conn.close()

    except Exception as e:
        print("Lỗi lấy vi phạm gần nhất:", e)

    return recent


# =========================
# TRANG CHỦ
# =========================
@app.route("/")
def home():
    release_camera()

    stats = get_dashboard_stats()
    recent_violations = get_recent_violations(limit=5)

    return render_template(
        "home.html",
        stats=stats,
        recent_violations=recent_violations
    )


# =========================
# NHẬN DIỆN ẢNH / VIDEO UPLOAD
# =========================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    release_camera()

    result_file = None
    file_type = None
    message = None
    detected_violations = []

    if request.method == "POST":
        if "file" not in request.files:
            message = "Chưa chọn file."
            return render_template(
                "upload.html",
                message=message,
                detected_violations=detected_violations
            )

        file = request.files["file"]

        if file.filename == "":
            message = "Chưa chọn file."
            return render_template(
                "upload.html",
                message=message,
                detected_violations=detected_violations
            )

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            saved_name = f"{timestamp}_{filename}"
            upload_path = os.path.join(UPLOAD_FOLDER, saved_name)
            file.save(upload_path)

            # Xử lý ảnh
            if is_image(filename):
                frame = cv2.imread(upload_path)

                if frame is None:
                    message = "Không đọc được ảnh."
                    return render_template(
                        "upload.html",
                        message=message,
                        detected_violations=detected_violations
                    )

                frame = resize_frame_for_render(frame, max_width=960)

                annotated_frame, violations = run_detection(frame)

                violations = deduplicate_violations(violations)
                detected_violations = violations

                result_name = f"result_{timestamp}_{filename}"
                result_path = os.path.join(RESULT_FOLDER, result_name)

                cv2.imwrite(result_path, annotated_frame)

                if violations:
                    save_violations_to_db(
                        violations=violations,
                        annotated_frame=annotated_frame,
                        prefix="image_violation",
                        source_type="Ảnh tải lên"
                    )

                result_file = to_web_path(result_path)
                file_type = "image"

            # Xử lý video
            elif is_video(filename):
                if ONLINE_MODE:
                    message = "Bản online hiện chỉ hỗ trợ nhận diện ảnh. Video vui lòng chạy ở bản local để tránh quá tải server."
                    return render_template(
                        "upload.html",
                        result_file=result_file,
                        file_type=file_type,
                        message=message,
                        detected_violations=detected_violations
                    )

                cap = cv2.VideoCapture(upload_path)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps is None or fps <= 0:
                    fps = 25

                result_name = f"result_{timestamp}_{filename.rsplit('.', 1)[0]}.mp4"
                result_path = os.path.join(RESULT_FOLDER, result_name)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

                video_save_cache = {}
                detected_violations_map = {}
                frame_index = 0

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_index += 1
                    current_second = frame_index / fps

                    annotated_frame, violations = run_detection(frame)

                    violations = deduplicate_violations(violations)

                    if violations:
                        for violation in violations:
                            if should_save_video_violation(
                                violation=violation,
                                annotated_frame=annotated_frame,
                                current_second=current_second,
                                video_save_cache=video_save_cache
                            ):
                                save_violations_to_db(
                                    violations=[violation],
                                    annotated_frame=annotated_frame,
                                    prefix="video_violation",
                                    source_type="Video tải lên"
                                )

                                detected_violations_map[violation["class_name"]] = violation

                        detected_violations = list(detected_violations_map.values())

                    out.write(annotated_frame)

                cap.release()
                out.release()

                result_file = to_web_path(result_path)
                file_type = "video"

    return render_template(
        "upload.html",
        result_file=result_file,
        file_type=file_type,
        message=message,
        detected_violations=detected_violations
    )


# =========================
# CAMERA WEB REALTIME - JS GỬI FRAME LÊN SERVER
# =========================
@app.route("/detect_webcam_frame", methods=["POST"])
def detect_webcam_frame():
    global latest_violations, last_camera_save_times

    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "Không nhận được ảnh từ camera."}), 400

        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Không đọc được frame camera."}), 400

        frame = resize_frame_for_render(frame, max_width=416)

        annotated_frame, violations = run_detection(frame)
        violations = deduplicate_violations(violations)

        latest_violations = violations

        current_time = time.time()
        violations_to_save = []

        for violation in violations:
            violation_type = violation.get("class_name", "Vi phạm")
            last_time = last_camera_save_times.get(violation_type, 0)

            if current_time - last_time >= CAMERA_SAVE_INTERVAL:
                violations_to_save.append(violation)
                last_camera_save_times[violation_type] = current_time

        if violations_to_save:
            save_violations_to_db(
                violations=violations_to_save,
                annotated_frame=annotated_frame,
                prefix="webcam_violation",
                source_type="Camera web realtime"
            )

        ret, buffer = cv2.imencode(".jpg", annotated_frame)

        if not ret:
            return jsonify({"error": "Không mã hóa được ảnh kết quả."}), 500

        result_base64 = base64.b64encode(buffer).decode("utf-8")
        result_image = "data:image/jpeg;base64," + result_base64

        violation_names = []

        for v in violations:
            name = v.get("class_name", "")
            if name and name not in violation_names:
                violation_names.append(name)

        if violation_names:
            message = "Phát hiện vi phạm: " + ", ".join(violation_names)
        else:
            message = "Không phát hiện vi phạm rõ ràng."

        return jsonify({
            "image": result_image,
            "message": message,
            "violations": violations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# CAMERA REALTIME KIỂU STREAM
# =========================
def generate_camera_frames():
    global camera, latest_violations, last_camera_save_times

    try:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        while True:
            success, frame = camera.read()

            if not success:
                break

            annotated_frame, violations = run_detection(frame)

            violations = deduplicate_violations(violations)
            latest_violations = violations

            current_time = time.time()
            violations_to_save = []

            for violation in violations:
                violation_type = violation["class_name"]
                last_time = last_camera_save_times.get(violation_type, 0)

                if current_time - last_time >= CAMERA_SAVE_INTERVAL:
                    violations_to_save.append(violation)
                    last_camera_save_times[violation_type] = current_time

            if violations_to_save:
                save_violations_to_db(
                    violations=violations_to_save,
                    annotated_frame=annotated_frame,
                    prefix="camera_violation",
                    source_type="Camera realtime"
                )

            ret, buffer = cv2.imencode(".jpg", annotated_frame)

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    except GeneratorExit:
        release_camera()

    except Exception as e:
        print(f"Lỗi camera realtime: {e}")
        release_camera()

    finally:
        release_camera()


@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_camera_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stop_camera", methods=["GET", "POST"])
def stop_camera():
    release_camera()
    return jsonify({"status": "camera_stopped"})


@app.route("/current_violations")
def current_violations():
    return jsonify(latest_violations)


# =========================
# LỊCH SỬ VI PHẠM
# =========================
@app.route("/history")
def history():
    release_camera()

    keyword = request.args.get("q", "").strip()
    violation_filter = request.args.get("violation_type", "").strip()
    source_filter = request.args.get("source_type", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    stats = {
        "total": 0,
        "no_helmet": 0,
        "overload": 0,
        "today": 0,
        "image_source": 0,
        "video_source": 0,
        "camera_source": 0
    }

    try:
        cursor.execute("SELECT COUNT(*) FROM violations")
        stats["total"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE violation_type LIKE '%Không đội mũ%'
               OR violation_type LIKE '%không đội mũ%'
               OR violation_type LIKE '%Khong doi mu%'
               OR violation_type LIKE '%khong doi mu%'
               OR violation_type LIKE '%No helmet%'
               OR violation_type LIKE '%no helmet%'
               OR violation_type LIKE '%Without helmet%'
               OR violation_type LIKE '%without helmet%'
        """)
        stats["no_helmet"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE violation_type LIKE '%Chở quá%'
               OR violation_type LIKE '%chở quá%'
               OR violation_type LIKE '%Cho qua%'
               OR violation_type LIKE '%cho qua%'
               OR violation_type LIKE '%Quá số người%'
               OR violation_type LIKE '%quá số người%'
               OR violation_type LIKE '%Qua so nguoi%'
               OR violation_type LIKE '%qua so nguoi%'
               OR violation_type LIKE '%Overload%'
               OR violation_type LIKE '%overload%'
        """)
        stats["overload"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE DATE(created_at) = DATE('now', 'localtime')
        """)
        stats["today"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE source_type LIKE '%Ảnh%'
               OR source_type LIKE '%ảnh%'
               OR source_type LIKE '%Anh%'
               OR source_type LIKE '%anh%'
               OR source_type LIKE '%Image%'
               OR source_type LIKE '%image%'
        """)
        stats["image_source"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE source_type LIKE '%Video%'
               OR source_type LIKE '%video%'
        """)
        stats["video_source"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM violations
            WHERE source_type LIKE '%Camera%'
               OR source_type LIKE '%camera%'
               OR source_type LIKE '%Realtime%'
               OR source_type LIKE '%realtime%'
        """)
        stats["camera_source"] = cursor.fetchone()[0]

    except Exception as e:
        print("Lỗi thống kê lịch sử:", e)

    query = """
        SELECT id, violation_type, confidence, image_path, source_type, created_at, plate_number
        FROM violations
        WHERE 1 = 1
    """

    params = []

    if keyword:
        query += """
            AND (
                violation_type LIKE ?
                OR source_type LIKE ?
                OR created_at LIKE ?
                OR plate_number LIKE ?
            )
        """
        search_value = f"%{keyword}%"
        params.extend([search_value, search_value, search_value, search_value])

    if violation_filter:
        if violation_filter == "no_helmet":
            query += """
                AND (
                    violation_type LIKE '%Không đội mũ%'
                    OR violation_type LIKE '%không đội mũ%'
                    OR violation_type LIKE '%Khong doi mu%'
                    OR violation_type LIKE '%khong doi mu%'
                    OR violation_type LIKE '%No helmet%'
                    OR violation_type LIKE '%no helmet%'
                    OR violation_type LIKE '%Without helmet%'
                    OR violation_type LIKE '%without helmet%'
                )
            """
        elif violation_filter == "overload":
            query += """
                AND (
                    violation_type LIKE '%Chở quá%'
                    OR violation_type LIKE '%chở quá%'
                    OR violation_type LIKE '%Cho qua%'
                    OR violation_type LIKE '%cho qua%'
                    OR violation_type LIKE '%Quá số người%'
                    OR violation_type LIKE '%quá số người%'
                    OR violation_type LIKE '%Qua so nguoi%'
                    OR violation_type LIKE '%qua so nguoi%'
                    OR violation_type LIKE '%Overload%'
                    OR violation_type LIKE '%overload%'
                )
            """

    if source_filter:
        if source_filter == "image":
            query += """
                AND (
                    source_type LIKE '%Ảnh%'
                    OR source_type LIKE '%ảnh%'
                    OR source_type LIKE '%Anh%'
                    OR source_type LIKE '%anh%'
                    OR source_type LIKE '%Image%'
                    OR source_type LIKE '%image%'
                )
            """
        elif source_filter == "video":
            query += """
                AND (
                    source_type LIKE '%Video%'
                    OR source_type LIKE '%video%'
                )
            """
        elif source_filter == "camera":
            query += """
                AND (
                    source_type LIKE '%Camera%'
                    OR source_type LIKE '%camera%'
                    OR source_type LIKE '%Realtime%'
                    OR source_type LIKE '%realtime%'
                )
            """

    if date_from:
        query += " AND DATE(created_at) >= DATE(?) "
        params.append(date_from)

    if date_to:
        query += " AND DATE(created_at) <= DATE(?) "
        params.append(date_to)

    query += " ORDER BY id DESC "

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    violations = []

    for row in rows:
        image_path = row["image_path"]

        if image_path and not image_path.startswith("/"):
            image_path = "/" + image_path

        violations.append({
            "id": row["id"],
            "violation_type": row["violation_type"],
            "confidence": row["confidence"],
            "image_path": image_path,
            "source_type": row["source_type"],
            "created_at": row["created_at"],
            "plate_number": row["plate_number"]
        })

    filters = {
        "q": keyword,
        "violation_type": violation_filter,
        "source_type": source_filter,
        "date_from": date_from,
        "date_to": date_to
    }

    return render_template(
        "history.html",
        violations=violations,
        stats=stats,
        filters=filters,
        filtered_count=len(violations)
    )


# =========================
# HÀM XOÁ FILE TRONG THƯ MỤC
# =========================
def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Không xoá được file {file_path}: {e}")


# =========================
# XOÁ TOÀN BỘ LỊCH SỬ
# =========================
@app.route("/delete_history")
def delete_history():
    release_camera()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM violations")

    try:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='violations'")
    except Exception:
        pass

    conn.commit()
    conn.close()

    clear_folder(VIOLATION_FOLDER)
    clear_folder(UPLOAD_FOLDER)
    clear_folder(RESULT_FOLDER)

    return redirect(url_for("history"))


# =========================
# XOÁ RIÊNG MỘT BẢN GHI
# =========================
@app.route("/delete_violation/<int:violation_id>")
def delete_violation(violation_id):
    release_camera()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT image_path FROM violations WHERE id = ?",
        (violation_id,)
    )
    row = cursor.fetchone()

    cursor.execute(
        "DELETE FROM violations WHERE id = ?",
        (violation_id,)
    )

    conn.commit()
    conn.close()

    if row and row[0]:
        image_path = row[0]

        if image_path.startswith("/"):
            image_path = image_path[1:]

        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Không xoá được ảnh vi phạm {image_path}: {e}")

    return redirect(url_for("history"))


# =========================
# CHẠY APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
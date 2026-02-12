# utils/ocr.py
import os
import json
import cv2
import numpy as np
import easyocr

# ------------------------------------------------------------
# EasyOCR INITIALIZATION (GLOBAL, ONCE)
# ------------------------------------------------------------
reader = easyocr.Reader(
    ['en', 'hi'],  # extend if needed
    gpu=False
)

# ------------------------------------------------------------
# OCR CACHE (RELATIVE, SUBMISSION SAFE)
# ------------------------------------------------------------
OCR_CACHE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ocr_cache"
)
os.makedirs(OCR_CACHE_DIR, exist_ok=True)


def get_ocr_cache_path(img_name):
    return os.path.join(
        OCR_CACHE_DIR,
        img_name.rsplit(".", 1)[0] + ".json"
    )


# ------------------------------------------------------------
# DESKEW (UNCHANGED LOGIC)
# ------------------------------------------------------------
def deskew_image(gray):
    inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(
        inv, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=200,
        minLineLength=1000,
        maxLineGap=20
    )

    if lines is None:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    return cv2.warpAffine(
        gray,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# ------------------------------------------------------------
# PREPROCESSING (UNCHANGED BEHAVIOR)
# ------------------------------------------------------------
def preprocess_document(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Orientation detection
    rotate_flag = False
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        angles = []
        for i in range(min(len(lines), 50)):
            rho, theta = lines[i][0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)

        if abs(np.median(angles)) > 45:
            rotate_flag = True

    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    # Resize (A4-ish)
    img = cv2.resize(img, (1654, 2339))
    gray = cv2.resize(gray, (1654, 2339))

    # Contrast normalize
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_norm = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray_norm, h=20)

    # Binary for OCR
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return denoised, binary_clean


# ------------------------------------------------------------
# STRUCTURED OCR BLOCKS
# ------------------------------------------------------------
def build_structured_ocr_blocks(ocr_raw):
    blocks = []
    for bbox, text, conf in ocr_raw:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]

        blocks.append({
            "text": text.strip(),
            "conf": float(conf),
            "bbox": {
                "x1": int(min(xs)),
                "y1": int(min(ys)),
                "x2": int(max(xs)),
                "y2": int(max(ys))
            }
        })
    return blocks


# ------------------------------------------------------------
# MAIN ENTRY: GET OCR BLOCKS (CACHED)
# ------------------------------------------------------------
def get_cached_ocr_blocks(img_path):
    img_name = os.path.basename(img_path)
    cache_path = get_ocr_cache_path(img_name)

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    _, binary = preprocess_document(img_path)
    ocr_raw = reader.readtext(binary)
    blocks = build_structured_ocr_blocks(ocr_raw)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)

    return blocks

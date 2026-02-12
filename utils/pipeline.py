# utils/pipeline.py

import os
import cv2

from utils.ocr import get_cached_ocr_blocks
from utils.extraction import *
from utils.yolo_detector import detect_stamp_and_signature
from utils.confidence import compute_overall_confidence


def process_invoice_single(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    H, W = img.shape[:2]

    # ---------- OCR ----------
    blocks = get_cached_ocr_blocks(img_path)

    # ---------- Rule-based extraction ----------
    dealer, dealer_conf = extract_dealer_name_with_conf(blocks, H, W)
    model, model_conf = extract_model_name_with_conf(blocks, H)
    hp, hp_conf = extract_horse_power_with_conf(blocks, model)
    asset_cost, asset_cost_conf = extract_asset_cost_with_conf(blocks, H, W)

    # ---------- YOLO stamp & signature ----------
    stamp_sig = detect_stamp_and_signature(img_path)

    # ---------- Overall confidence ----------
    overall_conf = compute_overall_confidence({
        "dealer_confidence": dealer_conf,
        "model_confidence": model_conf,
        "horse_power_confidence": hp_conf,
        "asset_cost_confidence": asset_cost_conf,
        "stamp_confidence": stamp_sig["stamp_confidence"],
        "signature_confidence": stamp_sig["signature_confidence"]
    })

    # ---------- Final JSON ----------
    return {
        "doc_id": os.path.basename(img_path).rsplit(".", 1)[0],

        "fields": {
            "dealer_name": {
                "value": dealer,
                "confidence": dealer_conf
            },
            "model_name": {
                "value": model,
                "confidence": model_conf
            },
            "horse_power": {
                "value": hp,
                "confidence": hp_conf
            },
            "asset_cost": {
                "value": asset_cost,
                "confidence": asset_cost_conf
            },
            "stamp": {
                "present": stamp_sig["stamp_present"],
                "bbox": stamp_sig["stamp_bbox"],
                "confidence": stamp_sig["stamp_confidence"]
            },
            "signature": {
                "present": stamp_sig["signature_present"],
                "bbox": stamp_sig["signature_bbox"],
                "confidence": stamp_sig["signature_confidence"]
            }
        },

        "overall_confidence": round(overall_conf, 3)
    }


def flatten_for_csv(image_name, result):
    f = result["fields"]
    return {
        "image_name": image_name,

        "dealer_name": f["dealer_name"]["value"],
        "dealer_confidence": f["dealer_name"]["confidence"],

        "model_name": f["model_name"]["value"],
        "model_confidence": f["model_name"]["confidence"],

        "horse_power": f["horse_power"]["value"],
        "horse_power_confidence": f["horse_power"]["confidence"],

        "asset_cost": f["asset_cost"]["value"],
        "asset_cost_confidence": f["asset_cost"]["confidence"],

        "stamp_present": f["stamp"]["present"],
        "stamp_confidence": f["stamp"]["confidence"],

        "signature_present": f["signature"]["present"],
        "signature_confidence": f["signature"]["confidence"],
    }

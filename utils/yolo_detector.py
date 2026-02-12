from ultralytics import YOLO
import os

# Load model ONCE when file is imported
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "stamp_signature_yolo.pt"
)

stamp_model = YOLO(MODEL_PATH)

def detect_stamp_and_signature(img_path, conf_thres=0.25):
    result = {
        "stamp_present": False,
        "stamp_bbox": None,
        "stamp_confidence": 0.0,
        "signature_present": False,
        "signature_bbox": None,
        "signature_confidence": 0.0
    }

    preds = stamp_model(img_path, conf=conf_thres, verbose=False)[0]

    for box, cls, conf in zip(
        preds.boxes.xyxy.cpu().numpy(),
        preds.boxes.cls.cpu().numpy(),
        preds.boxes.conf.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)

        if int(cls) == 1:  # stamp
            if conf > result["stamp_confidence"]:
                result["stamp_present"] = True
                result["stamp_bbox"] = [x1, y1, x2, y2]
                result["stamp_confidence"] = float(conf)

        elif int(cls) == 0:  # signature
            if conf > result["signature_confidence"]:
                result["signature_present"] = True
                result["signature_bbox"] = [x1, y1, x2, y2]
                result["signature_confidence"] = float(conf)

    return result

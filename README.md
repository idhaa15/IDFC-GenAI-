Invoice Information Extraction System
Hybrid Computer Vision + OCR Pipeline for Structured Invoice Understanding

Overview

This project implements a robust, production-oriented invoice processing pipeline that extracts structured information from scanned invoices, including:

Dealer Name

Model Name

Horse Power

Asset Cost

Stamp presence + bounding box

Signature presence + bounding box



The system is designed to work on real-world invoices with:

Multiple layouts

Mixed languages (English + Indian scripts)

Variable scan quality

Handwritten stamps and signatures



The final output is a structured JSON per invoice, along with:

Individual field confidence scores

An overall document confidence score

Optional annotated invoice images for explainability


Pipeline: 
Raw Invoice Image
        |
        v
[ Image Preprocessing ]
        |
        v
[ OCR (EasyOCR) ]
        |
        v
[ Structured OCR Blocks ]
        |
        +-----------------------------+
        |                             |
        v                             v
[ Rule-Based NLP Extraction ]   [ YOLOv8 Detection ]
        |                       (Stamp & Signature)
        |                             |
        +-------------+---------------+
                      |
                      v
              [ Confidence Fusion ]
                      |
                      v
               [ Final JSON Output ]



Image Preprocessing:
Goal: Improve OCR accuracy and normalize invoice scans.

Steps:

Orientation detection using Hough Lines

Rotation correction

A4-like resizing

Contrast normalization (CLAHE)

Noise removal

Adaptive thresholding for OCR


Why?

Invoices come scanned at:

Different angles

Different resolutions

Different lighting conditions

Preprocessing ensures consistent OCR performance.




OCR Engine
Selected OCR: EasyOCR

Languages enabled:

English, Hindi (Extendable to Marathi, Tamil, Telugu, etc.)

Why EasyOCR?

Strong multilingual support

Simple Python integration

Provides confidence scores per text block

Works well in CPU-only environments


OCR Output Format

Each OCR block is normalized into:
{
  "text": "ABC Tractors Pvt Ltd",
  "conf": 0.82,
  "bbox": {
    "x1": 120,
    "y1": 90,
    "x2": 820,
    "y2": 140
  }
}


OCR Caching (Latency Optimization)
Problem: OCR is expensive and slow when processing hundreds of invoices.

Solution: OCR results are cached on disk after the first run.
ocr_cache/
├── invoice_001.json
├── invoice_002.json
└── ...


On subsequent runs:

OCR is skipped
Cached structured blocks are reused

➡️ Massively reduces batch latency to ~20s per invoice. 


Rule-Based Field Extraction (NLP)

Instead of relying on black-box models, this system uses explainable heuristics.

Extracted Fields:

Dealer Name

Model Name

Horse Power

Asset Cost



Techniques Used

Regex patterns (multilingual-aware)

Position-based heuristics (top, center, right bias)

Character composition checks

Confidence-weighted scoring

Domain constraints (valid HP range, realistic cost range)


Stamp & Signature Detection (YOLOv8)
Why YOLO?

Stamps and signatures are visual objects, not text.

Model: YOLOv8 (custom-trained)

Dataset:

200+ invoices manually labeled

Bounding boxes for:
Stamp
Signature


"stamp": {
  "present": true,
  "bbox": [1136, 1901, 1439, 2230],
  "confidence": 0.95
}


Visual Explainability: Annotated invoices are saved with bounding boxes for submission and debugging.


Confidence Scoring
Per-Field Confidence

Each field includes:

Extraction confidence

Based on OCR quality + heuristic reliability




Overall Document Confidence computed as a weighted aggregation of:

Dealer confidence

Model confidence

Horse power confidence

Asset cost confidence

Stamp & signature confidence


Output JSON Format:
{
  "doc_id": "invoice_001",
  "fields": {
    "dealer_name": {
      "value": "ABC Tractors Pvt Ltd",
      "confidence": 0.92
    },
    "model_name": {
      "value": "Mahindra 575 DI",
      "confidence": 0.89
    },
    "horse_power": {
      "value": 50,
      "confidence": 0.85
    },
    "asset_cost": {
      "value": 525000,
      "confidence": 0.91
    },
    "signature": {
      "present": true,
      "bbox": [100, 200, 300, 250],
      "confidence": 0.78
    },
    "stamp": {
      "present": true,
      "bbox": [400, 500, 550, 600],
      "confidence": 0.94
    }
  },
  "overall_confidence": 0.96
}


Batch Processing

Processes hundreds of invoices in one run

Outputs:

One JSON per invoice

One CSV summary file

Annotated images for stamp/signature



Attempted Approaches (and Why They Were Not Used)
Vision–Language Models (VLMs)

Tested:

Qwen 2.5 / Qwen-VL (via Ollama / llama-cpp)

Issues encountered:

High latency

Inconsistent structured output

Difficult confidence calibration

Heavy resource usage

Less predictable behavior on varied invoice layouts

Decision:
Rule-based extraction + CV was more stable, explainable, and production-ready.


OCR Alternatives

Tested:

PaddleOCR

Tesseract

Final Choice: EasyOCR
Due to better multilingual support and simpler integration.


Submission Structure:
submission.zip
│
├── executable.py        # Entry point
├── requirements.txt     # Dependencies
├── README.md            # This file
├── utils/
│   ├── ocr.py
│   ├── extraction.py
│   ├── yolo_detector.py
│   ├── pipeline.py
│   └── confidence.py
│
└── sample_output/
    └── result.json


Output Artifacts & Drive Structure:
In addition to the submitted submission.zip, I've also provided full batch outputs via the shared Google Drive link for transparency, verification, and qualitative evaluation.

IDFC_Document_AI/
│
├── data/
│   ├── raw/
│   │   ├── 172684006_1_pg42.png
│   │   ├── 172717541_1_pg17.png
│   │   ├── ...
│   │   └── (All original invoice images)
│   │
│   └── processed/
│       ├── gray/
│       │   ├── 172684006_1_pg42.png
│       │   └── (Grayscale, deskewed images)
│       │
│       ├── binary/
│       │   ├── 172684006_1_pg42.png
│       │   └── (OCR-friendly binarized images)
│       │
│       ├── ocr_visuals/
│       │   ├── 172684006_1_pg42.png
│       │   └── (OCR bounding-box debug visuals)
│       │
│       └── ocr_cache/
│           ├── 172684006_1_pg42.json
│           └── (Cached structured OCR blocks)
│
├── json_outputs/
│   ├── 172684006_1_pg42.json
│   ├── 172717541_1_pg17.json
│   ├── ...
│   └── (Final JSON outputs for all 495 invoices)
│
├── visual_outputs/
│   └── stamp_signature/
│       ├── 172684006_1_pg42.png
│       ├── 172717541_1_pg17.png
│       ├── ...
│       └── (Invoices annotated with stamp & signature bboxes)




Future Improvements:

Expand multilingual normalization

Train larger YOLO dataset

Add layout-aware transformers (LayoutLM)

Ensemble OCR strategies

Automatic confidence calibration


Key Strengths of This Solution

✅ Explainable
✅ Deterministic
✅ Multilingual-ready
✅ Scalable
✅ Production-focused
✅ Hybrid CV + NLP
✅ Manually curated training data


Final Note

This system prioritizes reliability, transparency, and real-world usability over experimental complexity — making it suitable for deployment at scale.
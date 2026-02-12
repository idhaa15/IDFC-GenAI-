import re

DEALER_STOPWORDS = [
    "quotation", "invoice", "estimate", "gst", "date",
    "phone", "mobile", "email", "authorized", "dealer",
    "bank", "address"
]

def extract_dealer_name_with_conf(blocks, H, W):
    candidates = []

    for b in blocks:
        text = b["text"]
        conf = b["conf"]
        bb = b["bbox"]

        y_top = bb["y1"]
        width = bb["x2"] - bb["x1"]

        lower = text.lower()

        # --- filters ---
        if y_top > 0.35 * H:
            continue
        if width < 0.4 * W:
            continue
        if any(sw in lower for sw in DEALER_STOPWORDS):
            continue
        if sum(c.isalpha() for c in text) < 5:
            continue

        # --- score ---
        score = (
            0.4 * conf +
            0.3 * (width / W) +
            0.3 * (1 - y_top / H)
        )

        candidates.append((text, score))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_text, best_score = candidates[0]

    return best_text, round(min(best_score, 1.0), 3)


import re

MODEL_HINTS = ["model", "tractor", "rx", "di", "hp", "power"]

DIMENSION_REGEX = re.compile(r"\d+\s*['xX×]\s*\d+")
BAD_MODEL_REGEX = re.compile(r"^\d+$|^[A-Z]{1,2}$")

OCR_FIXES = {
    "Il": "II",
    "l0": "10",
    "O": "0",
    "S°": "50",
}

def normalize_model_text(text):
    out = text.strip()
    for k, v in OCR_FIXES.items():
        out = out.replace(k, v)
    out = re.sub(r"\s{2,}", " ", out)
    return out

def is_valid_model(text):
    if not text or len(text) < 4:
        return False
    if DIMENSION_REGEX.search(text):
        return False
    if BAD_MODEL_REGEX.match(text):
        return False
    if not any(c.isdigit() for c in text):
        return False
    if not any(c.isalpha() for c in text):
        return False
    return True


def extract_model_name_with_conf(blocks, H):
    candidates = []

    for b in blocks:
        text = b["text"]
        conf = b["conf"]
        bb = b["bbox"]

        y_center = (bb["y1"] + bb["y2"]) / 2
        lower = text.lower()

        # ---------- Position filter ----------
        if not (0.25 * H < y_center < 0.75 * H):
            continue

        # ---------- Must contain letters + digits ----------
        if not (any(c.isdigit() for c in text) and any(c.isalpha() for c in text)):
            continue

        # ---------- Scoring ----------
        hint_bonus = sum(h in lower for h in MODEL_HINTS) * 0.15

        score = (
            0.5 * conf +
            0.3 * hint_bonus +
            0.2 * (1 - abs(0.5 - y_center / H))
        )

        candidates.append((text, score))

    if not candidates:
        return None, 0.0

    # ---------- Pick best ----------
    candidates.sort(key=lambda x: x[1], reverse=True)
    raw_model, raw_score = candidates[0]

    # ================== POST-PROCESSING ==================
    model = normalize_model_text(raw_model)

    if not is_valid_model(model):
        return None, 0.0

    # ---------- Confidence calibration ----------
    conf = min(raw_score, 1.0)

    # boost if looks like real tractor model
    if re.search(r"(mf|rx|di|xt|hp|tractor)", model.lower()):
        conf = min(1.0, conf + 0.15)

    # penalize very short or noisy strings
    if len(model) < 5:
        conf *= 0.7

    return model, round(conf, 3)


import re

HP_REGEX = re.compile(
    r"""
    (?<!\d)
    (\d{2,3})
    \s*
    (?:hp|एचपी|एच\.पी\.|horse\s*power|ps)
    """,
    re.IGNORECASE | re.VERBOSE
)

MODEL_HP_REGEX = re.compile(
    r"(?<!\d)(\d{2,3})(?!\d)"
)
def extract_horse_power_with_conf(blocks, model_name=None):
    candidates = []

    # ---------- Pass 1: OCR text ----------
    for b in blocks:
        text = b["text"]
        conf = b["conf"]

        m = HP_REGEX.search(text)
        if not m:
            continue

        hp_val = int(m.group(1))

        # ---------- Hard validation ----------
        if not (20 <= hp_val <= 120):
            continue

        score = 0.65 * conf

        # boost if explicit HP keyword
        if re.search(r"(hp|horse|एच)", text.lower()):
            score += 0.25

        candidates.append((hp_val, score))

    # ---------- Pass 2: Model name fallback ----------
    if model_name:
        for m in MODEL_HP_REGEX.finditer(model_name):
            hp_val = int(m.group(1))
            if not (20 <= hp_val <= 120):
                continue

            # model-based inference is weaker but useful
            score = 0.55
            candidates.append((hp_val, score))

    if not candidates:
        return None, 0.0

    # ---------- Pick best ----------
    candidates.sort(key=lambda x: x[1], reverse=True)
    hp, score = candidates[0]

    # ---------- Final calibration ----------
    score = min(score, 1.0)

    if score < 0.6:
        return None, 0.0

    return hp, round(score, 3)


import re

MONEY_REGEX = re.compile(
    r"(?:₹|rs\.?|inr)?\s*([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.\d{1,2})?|[0-9]{5,9})",
    re.IGNORECASE
)


def extract_asset_cost_with_conf(blocks, H, W):
    candidates = []

    for b in blocks:
        text = b["text"]
        conf = b["conf"]
        bb = b["bbox"]

        lower = text.lower()

        # ❌ Exclude words-based totals
        if "only" in lower and not any(c.isdigit() for c in text):
            continue


        match = MONEY_REGEX.search(text)
        if not match:
            continue

        # Parse number safely
        raw_amt = match.group(1).replace(",", "")
        try:
            amount = int(float(raw_amt))
        except:
            continue

        # ❌ Filter unrealistic values
        if not (50000 <= amount <= 20000000):
            continue

        # Position heuristics
        x_center = (bb["x1"] + bb["x2"]) / 2
        y_center = (bb["y1"] + bb["y2"]) / 2

        right_bias = x_center / W          # higher = more right
        mid_bias = 1 - abs(0.6 - y_center / H)

        # Confidence score
        score = (
            0.45 * conf +
            0.35 * right_bias +
            0.20 * mid_bias
        )

        candidates.append((amount, score))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[1], reverse=True)
    amt, score = candidates[0]

    return amt, round(min(score, 1.0), 3)



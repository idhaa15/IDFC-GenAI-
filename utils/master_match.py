from rapidfuzz import fuzz, process

# TODO: replace with your real master file (CSV/DB load) before submission.
# These are placeholders so the matching logic has something to run against.
DEALER_MASTER = [
    "ABC Tractors Pvt Ltd",
    "Shree Ganesh Motors",
    "Patel Agro Equipments",
]

MODEL_MASTER = [
    "Mahindra 575 DI",
    "John Deere 5050 D",
    "Swaraj 744 FE",
]


def match_dealer_name(candidate, threshold=90):
    """Fuzzy match per PS spec: dealer name uses >=90% fuzzy match."""
    if not candidate:
        return None, 0.0

    best_match, score, _ = process.extractOne(
        candidate, DEALER_MASTER, scorer=fuzz.token_sort_ratio
    )

    if score >= threshold:
        return best_match, round(score / 100, 3)

    # No confident master match — keep raw OCR text but signal low trust
    return candidate, 0.0


def match_model_name(candidate, threshold=95):
    """Model name is exact-match per PS spec; threshold stays high to
    only absorb OCR noise (e.g. 'MahIndra' vs 'Mahindra'), not genuine
    ambiguity between different models."""
    if not candidate:
        return None, 0.0

    best_match, score, _ = process.extractOne(
        candidate, MODEL_MASTER, scorer=fuzz.ratio
    )

    if score >= threshold:
        return best_match, round(score / 100, 3)

    return candidate, 0.0

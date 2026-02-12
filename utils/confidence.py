def compute_overall_confidence(result):
    weights = {
        "dealer": 0.20,
        "model": 0.20,
        "hp": 0.20,
        "asset": 0.20,
        "visual": 0.20
    }

    visual_conf = max(
        result.get("stamp_confidence", 0),
        result.get("signature_confidence", 0)
    )

    score = (
        weights["dealer"] * result.get("dealer_confidence", 0) +
        weights["model"]  * result.get("model_confidence", 0) +
        weights["hp"]     * result.get("horse_power_confidence", 0) +
        weights["asset"]  * result.get("asset_cost_confidence", 0) +
        weights["visual"] * visual_conf
    )

    return round(score, 3)

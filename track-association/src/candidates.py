"""Candidate generation: cheap physical-possibility filter shared by both scorers.

Settings are derived from TRAIN true-link distributions by scripts/train.py, checked on
VAL and then frozen in artifacts/config.json. Labels are never consulted here.
"""
from src.features import pair_features


def passes(f, settings):
    if f["gap_min"] > settings["max_gap_min"]:
        return False
    if f["implied_speed_kmh"] > settings["max_implied_speed_kmh"]:
        return False
    best_err = min(f["fwd_err_km"], f["back_err_km"])
    if best_err > settings["max_err_km_per_min"] * (settings["err_offset_min"] + f["gap_min"]):
        return False
    return True


def generate_candidates(summaries, settings):
    """summaries: tracklet summaries of ONE scenario. Returns sorted [(a_id, b_id, features)]."""
    out = []
    ordered = sorted(summaries, key=lambda s: s["tracklet_id"])
    for a in ordered:
        for b in ordered:
            if a["tracklet_id"] == b["tracklet_id"]:
                continue
            if b["start_time"] <= a["end_time"]:  # B must start strictly after A ends
                continue
            f = pair_features(a, b)
            if passes(f, settings):
                out.append((a["tracklet_id"], b["tracklet_id"], f))
    return out

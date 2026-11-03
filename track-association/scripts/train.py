"""Fit on TRAIN, select on VAL, save artifacts. Never reads test.

    python -m scripts.train
Outputs: artifacts/model.joblib, artifacts/config.json, results/validation_sweep.json
"""
import itertools
import json
import math
from pathlib import Path

import joblib
import numpy as np

from src.evaluate import load_labels, metrics
from src.features import ALL_FEATURES, PHYSICS_FEATURES, pair_features
from src.pipeline import load_folder, prepare
from src.decide import decide
from src.scorers import SEED, baseline_score, learned_scores, train_model

ROOT = Path(__file__).resolve().parents[1]
DATA, ART, RES = ROOT / "data", ROOT / "artifacts", ROOT / "results"
MODEL_VERSION = "v1.0.0"
MIN_PRECISION = 0.98          # wrong links are costlier than unresolved ones: protect precision first
ERR_OFFSET_MIN = 2.0
OPEN = {"max_gap_min": 1e9, "max_implied_speed_kmh": 1e9, "max_err_km_per_min": 1e9, "err_offset_min": ERR_OFFSET_MIN}

COST_WRONG = 5               # one wrong link is treated as costing as much as 5 missed links

BASE_T = [-0.02, -0.05, -0.1, -0.15, -0.2, -0.3, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, -5.0, -12.0]
BASE_M = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
BASE_T_KM = [-0.5, -1.0, -2.0, -4.0, -8.0, -12.0, -20.0, -40.0, -80.0]
BASE_M_KM = [0.0, 0.5, 1.0, 2.0, 4.0]
LEARN_T = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
LEARN_M = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def derive_candidate_settings(prepared, labels):
    """Look at TRAIN true links only to set the physical-possibility gates (~99.5 % retention)."""
    gaps, speeds, ratios = [], [], []
    for _, summaries, _ in prepared:
        by_id = {s["tracklet_id"]: s for s in summaries}
        for a in by_id:
            b = labels.get(a)
            if b and b in by_id:
                f = pair_features(by_id[a], by_id[b])
                gaps.append(f["gap_min"]); speeds.append(f["implied_speed_kmh"])
                ratios.append(min(f["fwd_err_km"], f["back_err_km"]) / (ERR_OFFSET_MIN + f["gap_min"]))
    s = {
        "max_gap_min": float(math.ceil(np.percentile(gaps, 99.9) / 5) * 5),
        "max_implied_speed_kmh": float(math.ceil(np.percentile(speeds, 99.9) * 1.1 / 10) * 10),
        "max_err_km_per_min": round(float(np.percentile(ratios, 99.5)) * 1.1, 4),
        "err_offset_min": ERR_OFFSET_MIN,
    }
    s["derived_from"] = {"n_train_true_links": len(gaps), "gap_p99.9": round(float(np.percentile(gaps, 99.9)), 2),
                         "implied_speed_p99.9": round(float(np.percentile(speeds, 99.9)), 1),
                         "err_ratio_p99.5": round(float(np.percentile(ratios, 99.5)), 4)}
    return s


def sweep(prepared, labels, score_fn, thresholds, margins, extra=None):
    scored = [(ss, c, score_fn(c)) for _, ss, c in prepared]
    all_c = [x for _, _, c in prepared for x in c]
    rows = []
    for t, m, sk in itertools.product(thresholds, margins, [False, True]):
        res = {}
        for ss, c, sc in scored:
            res.update(decide([s["tracklet_id"] for s in ss], c, sc, t, m, skip_check=sk))
        rows.append({**(extra or {}), "threshold": t, "margin": m, "skip_check": sk, **metrics(res, labels, all_c)})
    return rows


def utility(r):
    return r["correct_links"] - COST_WRONG * r["incorrect_links"]


def pick(rows):
    """Max utility (correct - COST_WRONG * incorrect) subject to precision >= MIN_PRECISION;
    ties -> more conservative (higher threshold, then higher margin)."""
    for r in rows:
        r["utility"] = utility(r)
    ok = [r for r in rows if r["precision"] is not None and r["precision"] >= MIN_PRECISION]
    if not ok:
        raise SystemExit("no setting reaches the precision floor on validation")
    return max(ok, key=lambda r: (r["utility"], r["threshold"], r["margin"], r["skip_check"]))


def reliability(model, prepared, labels, features, bins=10):
    """Observed link rate per score bin on VAL candidates (descriptive; no calibration claim)."""
    s, y = [], []
    for _, _, c in prepared:
        s += learned_scores(model, c, features)
        y += [1 if labels.get(a) == b else 0 for a, b, _ in c]
    s, y = np.array(s), np.array(y)
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (s >= lo) & (s < hi if i < bins - 1 else s <= hi)
        out.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                    "mean_score": round(float(s[m].mean()), 3) if m.any() else None,
                    "observed_true_rate": round(float(y[m].mean()), 3) if m.any() else None})
    return out


def main():
    ART.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    train_payload = load_folder(DATA / "train")
    train_labels = load_labels(DATA / "train" / "labels.jsonl")
    _, train_open, _, _ = prepare(train_payload, OPEN | {"max_gap_min": 240})
    cand = derive_candidate_settings(train_open, train_labels)
    print("candidate settings:", json.dumps(cand))
    gates = {k: v for k, v in cand.items() if k != "derived_from"}

    _, train_prep, _, _ = prepare(train_payload, gates)
    val_labels = load_labels(DATA / "val" / "labels.jsonl")
    _, val_prep, _, _ = prepare(load_folder(DATA / "val"), gates)
    train_c = [c for _, _, cs in train_prep for c in cs]
    val_c = [c for _, _, cs in val_prep for c in cs]
    cand_recall = {"train": metrics({}, train_labels, train_c)["candidate_recall"],
                   "val": metrics({}, val_labels, val_c)["candidate_recall"]}
    print("candidate recall:", cand_recall)

    # baseline
    base_rows = (sweep(val_prep, val_labels, lambda c: [baseline_score(f, "per_minute") for _, _, f in c],
                       BASE_T, BASE_M, {"variant": "per_minute"})
                 + sweep(val_prep, val_labels, lambda c: [baseline_score(f, "km") for _, _, f in c],
                         BASE_T_KM, BASE_M_KM, {"variant": "km"}))
    base_best = pick(base_rows)
    print("baseline best on val:", {k: base_best[k] for k in ("variant", "threshold", "margin", "skip_check", "precision", "recall")})

    # learned: feature-set selection on validation
    feature_sets = {"physics": PHYSICS_FEATURES, "physics+supporting": ALL_FEATURES}
    learned = {}
    for name, feats in feature_sets.items():
        model = train_model(train_c, train_labels, feats, SEED)
        rows = sweep(val_prep, val_labels, lambda c, m=model, f=feats: learned_scores(m, c, f), LEARN_T, LEARN_M)
        learned[name] = {"model": model, "features": feats, "rows": rows, "best": pick(rows)}
        print(f"learned[{name}] best on val:", {k: learned[name]["best"][k] for k in ("threshold", "margin", "skip_check", "precision", "recall")})
    chosen = max(learned, key=lambda n: (learned[n]["best"]["utility"], -len(learned[n]["features"])))
    L = learned[chosen]

    config = {
        "model_version": MODEL_VERSION,
        "seed": SEED,
        "model": "sklearn HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)",
        "feature_set": chosen,
        "features": L["features"],
        "candidates": gates,
        "candidate_settings_derivation": cand["derived_from"],
        "baseline": {"variant": base_best["variant"], "threshold": base_best["threshold"], "margin": base_best["margin"],
                     "skip_check": base_best["skip_check"],
                     "score": ("-mean(fwd_err_km, back_err_km)/(2+gap_min), km per gap-minute" if base_best["variant"] == "per_minute"
                               else "-mean(fwd_err_km, back_err_km), km") + "; higher is better; not a probability"},
        "learned": {"threshold": L["best"]["threshold"], "margin": L["best"]["margin"], "skip_check": L["best"]["skip_check"],
                    "score": "classifier output in [0,1], higher is better, not calibrated"},
        "selection_rule": f"validation: maximise correct - {COST_WRONG}*incorrect subject to precision >= {MIN_PRECISION}; ties -> higher threshold, higher margin",
        "trained_on": "data/train", "selected_on": "data/val",
    }
    joblib.dump(L["model"], ART / "model.joblib")
    (ART / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (RES / "validation_sweep.json").write_text(json.dumps({
        "candidate_recall": cand_recall,
        "baseline": {"chosen": base_best, "rows": base_rows},
        "learned": {n: {"features": v["features"], "chosen": v["best"], "rows": v["rows"]} for n, v in learned.items()},
        "learned_feature_set_chosen": chosen,
        "learned_reliability_val": reliability(L["model"], val_prep, val_labels, L["features"]),
    }, indent=2), encoding="utf-8")
    print(json.dumps({k: config[k] for k in ("feature_set", "candidates", "baseline", "learned")}, indent=2))


if __name__ == "__main__":
    main()

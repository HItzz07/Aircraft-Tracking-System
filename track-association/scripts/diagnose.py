"""Development diagnostic on VAL: categorise wrong accepted links of a scorer at a setting.

    python -m scripts.diagnose baseline -0.1 0.0
Categories use data/_truth/val.jsonl (debug-only ground truth; val is the selection split).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import joblib

from src.decide import decide
from src.evaluate import load_labels
from src.load import read_jsonl
from src.pipeline import load_folder, prepare
from src.scorers import baseline_score, learned_scores

ROOT = Path(__file__).resolve().parents[1]


def main(method, threshold, margin, part="val"):
    cfg = json.loads((ROOT / "artifacts" / "config.json").read_text())
    labels = load_labels(ROOT / "data" / part / "labels.jsonl")
    truth = {r["tracklet_id"]: r for r in read_jsonl(ROOT / "data" / "_truth" / f"{part}.jsonl")}
    _, prep, _, _ = prepare(load_folder(ROOT / "data" / part), cfg["candidates"])
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    cats = Counter()
    examples = []
    for _, ss, c in prep:
        sc = [baseline_score(f) for _, _, f in c] if method == "baseline" else learned_scores(model, c, cfg["features"])
        res = decide([s["tracklet_id"] for s in ss], c, sc, threshold, margin)
        for a, r in res.items():
            b = r["successor"]
            if not b or labels.get(a) == b:
                continue
            same_ac = truth[a]["aircraft_id"] == truth[b]["aircraft_id"]
            kind = ("skip (same aircraft, intermediate exists)" if same_ac else
                    "parallel partner" if truth[a]["follower"] or truth[b]["follower"] else "different aircraft")
            if labels.get(a) is None and not same_ac:
                kind += " / A has no successor"
            cats[kind] += 1
            m = r["ranked_candidates"][0]["measurements"]
            examples.append({"a": a, "b": b, "label": labels.get(a), "kind": kind, "score": r["score"],
                             "gap": m["gap_min"], "fwd": m["fwd_err_km"], "back": m["back_err_km"]})
    print(dict(cats))
    for e in examples[:12]:
        print(e)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), *(sys.argv[4:5]))

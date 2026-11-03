"""Frozen evaluation on TEST + stress tests (no retraining, no retuning).

    python -m scripts.evaluate
Outputs: results/test_metrics.json, results/test_errors.json
"""
import json
import random
from pathlib import Path

import joblib

from src.evaluate import load_labels, metrics
from src.load import read_jsonl
from src.pipeline import load_folder, run_association

ROOT = Path(__file__).resolve().parents[1]
DATA, ART, RES = ROOT / "data", ROOT / "artifacts", ROOT / "results"


def thin(payload, keep_every=3):
    """S1: keep every 3rd report per tracklet (by observed_at), always first + last.
    Labels unchanged; every tracklet keeps >= 2 points."""
    by_t = {}
    for r in payload["observations"]:
        by_t.setdefault(r["tracklet_id"], {})[r["observation_id"]] = r   # also drops duplicates
    kept = []
    for recs in by_t.values():
        recs = sorted(recs.values(), key=lambda r: (r["observed_at"], r["observation_id"]))
        for i, r in enumerate(recs):
            if i % keep_every == 0 or i == len(recs) - 1:
                kept.append(r)
    ids = {r["observation_id"] for r in kept}
    sup = dict(payload["supporting"])
    sup["id_reports"] = [r for r in sup["id_reports"] if r["observation_id"] in ids]
    return dict(payload, observations=kept, supporting=sup)


def drop_stream(payload, *streams):
    sup = {k: ([] if k in streams else v) for k, v in payload["supporting"].items()}
    return dict(payload, supporting=sup)


def noisy(payload, sigma_m=300, seed=7):
    """S3: add extra Gaussian position noise (sigma 300 m) to every report."""
    rng = random.Random(seed)
    out = []
    for r in payload["observations"]:
        r = dict(r)
        r["latitude"] = max(-90, min(90, r["latitude"] + rng.gauss(0, sigma_m) / 111_000))
        r["longitude"] = r["longitude"] + rng.gauss(0, sigma_m) / 75_000
        out.append(r)
    # identical duplicates would now conflict -> dedupe by id first (keep first)
    seen, uniq = set(), []
    for r in out:
        if r["observation_id"] not in seen:
            seen.add(r["observation_id"]); uniq.append(r)
    return dict(payload, observations=uniq)


def evaluate(payload, labels, model, config, name):
    resp, cands = run_association(payload, model, config, snapshot_id=name, include_tracklets=False)
    return resp, {"dataset": name, "manifest": resp["manifest"],
                  "baseline": metrics(resp["methods"]["baseline"], labels, cands),
                  "learned": metrics(resp["methods"]["learned"], labels, cands)}


def error_dump(resp, labels, truth):
    """Wrong links, missed links and disagreements, with ground-truth categories for analysis."""
    rows = []
    B, L = resp["methods"]["baseline"], resp["methods"]["learned"]
    for tid in sorted(labels):
        lab = labels[tid]
        entry = {"tracklet": tid, "label": lab, "aircraft": truth[tid]["aircraft_id"],
                 "baseline": {k: B[tid][k] for k in ("status", "successor", "score", "reason")},
                 "learned": {k: L[tid][k] for k in ("status", "successor", "score", "reason")},
                 "baseline_top": B[tid]["ranked_candidates"][:3], "learned_top": L[tid]["ranked_candidates"][:3]}
        tags = []
        for m, R in (("baseline", B), ("learned", L)):
            s = R[tid]["successor"]
            if s and s != lab:
                same = truth[s]["aircraft_id"] == truth[tid]["aircraft_id"]
                tags.append(f"{m}:wrong_link" + ("(skip)" if same else "(other aircraft)"))
            if lab and s is None:
                tags.append(f"{m}:missed")
        if B[tid]["successor"] != L[tid]["successor"]:
            tags.append("disagree")
        if tags:
            entry["tags"] = tags
            rows.append(entry)
    return rows


def main():
    model = joblib.load(ART / "model.joblib")
    config = json.loads((ART / "config.json").read_text())
    payload = load_folder(DATA / "test")
    labels = load_labels(DATA / "test" / "labels.jsonl")
    truth = {r["tracklet_id"]: r for r in read_jsonl(DATA / "_truth" / "test.jsonl")}

    resp, main_report = evaluate(payload, labels, model, config, "test")
    stress = {
        "S1_thinned_every_3rd": evaluate(thin(payload), labels, model, config, "test_thinned")[1],
        "S2_no_id_reports": evaluate(drop_stream(payload, "id_reports"), labels, model, config, "test_no_id_reports")[1],
        "S2b_no_supporting_streams": evaluate(drop_stream(payload, "id_reports", "tracklet_context"), labels, model, config,
                                              "test_no_supporting")[1],
        "S3_extra_noise_300m": evaluate(noisy(payload), labels, model, config, "test_noise300")[1],
    }
    sweep = json.loads((RES / "validation_sweep.json").read_text())
    out = {
        "note": "Test evaluated once at the frozen operating settings from artifacts/config.json. "
                "Threshold comparison below is from VALIDATION (used for selection), per the brief.",
        "frozen_config": {k: config[k] for k in ("model_version", "feature_set", "candidates", "baseline", "learned")},
        "test": main_report,
        "stress_tests": stress,
        "validation_threshold_comparison": threshold_table(sweep, config),
    }
    (RES / "test_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    errors = error_dump(resp, labels, truth)
    (RES / "test_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print_table(out)
    print(f"\n{len(errors)} tracklets with an error or disagreement written to results/test_errors.json")


def threshold_table(sweep, config):
    """Conservative / chosen / permissive rows from the validation sweep at the chosen margin & variant."""
    def rows_for(rows, cfg, keys):
        same = [r for r in rows if all(r.get(k) == cfg.get(k) for k in keys)]
        same.sort(key=lambda r: r["threshold"])
        ts = [r["threshold"] for r in same]
        i = ts.index(cfg["threshold"])
        pick = {"permissive": same[max(0, i - 2)], "chosen": same[i], "conservative": same[min(len(same) - 1, i + 2)]}
        return {k: {f: v[f] for f in ("threshold", "margin", "precision", "recall", "incorrect_links",
                                       "unresolved_proportion", "accepted_links")} for k, v in pick.items()}
    return {
        "baseline": rows_for(sweep["baseline"]["rows"], config["baseline"], ("variant", "margin", "skip_check")),
        "learned": rows_for(sweep["learned"][config["feature_set"]]["rows"], config["learned"], ("margin", "skip_check")),
    }


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def print_table(out):
    def line(name, m):
        print(f"{name:<34} P={fmt(m['precision'])} R={fmt(m['recall'])} wrong={m['incorrect_links']:<3} "
              f"unresolved={fmt(m['unresolved_proportion'])} cand_recall={fmt(m['candidate_recall'])} "
              f"accepted={m['accepted_links']}/{m['labeled_links']}")
    for method in ("baseline", "learned"):
        line(f"test {method}", out["test"][method])
    for s, rep in out["stress_tests"].items():
        for method in ("baseline", "learned"):
            line(f"{s} {method}", rep[method])


if __name__ == "__main__":
    main()

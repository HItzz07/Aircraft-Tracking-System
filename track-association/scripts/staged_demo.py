"""Staged additional-evidence demonstration.

Snapshot 1 and snapshot 2 are analysed INDEPENDENTLY (each is a complete input holding
only records received by its cutoff) with the SAME fitted model and decision settings.

    python -m scripts.staged_demo
Outputs: results/stage_1.json, results/stage_2.json, results/staged_diff.json
"""
import json
from pathlib import Path

import joblib

from src.evaluate import load_labels, metrics
from src.pipeline import load_folder, run_association

ROOT = Path(__file__).resolve().parents[1]
STAGED, ART, RES = ROOT / "data" / "staged", ROOT / "artifacts", ROOT / "results"
KEYS = ("gap_min", "dist_km", "fwd_err_km", "back_err_km", "heading_change_deg", "alt_diff_ft", "callsign_match")


def run_stage(snap, model, config):
    folder = STAGED / snap["folder"]
    payload = load_folder(folder)
    resp, cands = run_association(payload, model, config, snapshot_id=snap["snapshot_id"])
    labels = load_labels(folder / "labels.jsonl")
    resp["manifest"]["input_folder"] = str(folder.relative_to(ROOT)).replace("\\", "/")
    resp["manifest"]["received_cutoff"] = snap["received_cutoff"]
    resp["metrics"] = {m: metrics(resp["methods"][m], labels, cands) for m in ("baseline", "learned")}
    resp["labels"] = labels   # evaluation display only; never used by the pipeline
    (RES / f"{snap['snapshot_id']}.json").write_text(json.dumps(resp, indent=2, default=str), encoding="utf-8")
    return resp


def measurements(res, a, b):
    for c in res[a]["ranked_candidates"]:
        if c["candidate"] == b:
            return {k: c["measurements"].get(k) for k in KEYS} | {"score": c["score"]}
    return None


def diff(s1, s2, method):
    r1, r2 = s1["methods"][method], s2["methods"][method]
    t1, t2 = s1["tracklets"], s2["tracklets"]
    changes = []
    for tid in sorted(set(r1) | set(r2)):
        if tid not in r1:
            if r2[tid]["predecessor"] or r2[tid]["successor"]:
                changes.append({"tracklet": tid, "change": "new tracklet in stage 2",
                                "stage_2": {"successor": r2[tid]["successor"], "predecessor": r2[tid]["predecessor"]},
                                "n_points": t2[tid]["n_points"], "reason_stage_2": r2[tid]["reason"],
                                "explanation": "fragment received after the stage-1 cutoff (delayed tracklet)"})
            continue
        before, after = r1[tid]["successor"], r2[tid]["successor"]
        extended = t1[tid]["n_points"] != t2[tid]["n_points"]
        if before == after and not extended:
            continue
        if before == after:
            kind = "unchanged decision, tracklet extended by delayed reports"
        elif before is None:
            kind = "new link"
        elif after is None:
            kind = "withdrawn"
        else:
            kind = "replaced"
        expl = ""
        if kind in ("replaced", "withdrawn") and after is not None and after not in r1 and r2[after]["successor"] == before:
            expl = (f"{after} arrived late and sits inside the {tid}->{before} gap; the chain is now {tid}->{after}->{before}. "
                    f"This is the same inferred aircraft with a newly available intermediate fragment, not a change of identity.")
        elif kind == "withdrawn" and before in r2 and r2[before]["predecessor"] not in (None, tid):
            expl = f"{before} is now explained by {r2[before]['predecessor']} (a newly available or better-fitting predecessor)."
        elif kind == "new link" and after not in r1:
            expl = f"{after} was not yet received at the stage-1 cutoff; it is a delayed fragment."
        elif kind == "new link":
            expl = "new evidence (more reports / supporting records) raised the score or removed an ambiguity."
        elif extended:
            expl = f"delayed reports extended {tid} from {t1[tid]['n_points']} to {t2[tid]['n_points']} points; end time {t1[tid]['end_time']} -> {t2[tid]['end_time']}."
        changes.append({
            "tracklet": tid, "change": kind,
            "stage_1": {"status": r1[tid]["status"], "successor": before, "reason": r1[tid]["reason"],
                        "measurements": measurements(r1, tid, before) if before else None},
            "stage_2": {"status": r2[tid]["status"], "successor": after, "reason": r2[tid]["reason"],
                        "measurements": measurements(r2, tid, after) if after else None},
            "explanation": expl,
        })
    return changes


def main():
    model = joblib.load(ART / "model.joblib")
    config = json.loads((ART / "config.json").read_text())
    snaps = json.loads((STAGED / "snapshots.json").read_text())["snapshots"]
    s1, s2 = (run_stage(s, model, config) for s in snaps)
    summary = {
        "stage_1": s1["manifest"] | {"metrics": s1["metrics"]},
        "stage_2": s2["manifest"] | {"metrics": s2["metrics"]},
        "stage_1_accepted": {m: {t: r["successor"] for t, r in s1["methods"][m].items() if r["successor"]} for m in ("baseline", "learned")},
        "stage_1_unresolved": {m: sorted(t for t, r in s1["methods"][m].items() if not r["successor"]) for m in ("baseline", "learned")},
        "changes": {m: diff(s1, s2, m) for m in ("baseline", "learned")},
    }
    (RES / "staged_diff.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for m in ("baseline", "learned"):
        print(f"\n== {m}: stage1 P={s1['metrics'][m]['precision']} R={s1['metrics'][m]['recall']} | "
              f"stage2 P={s2['metrics'][m]['precision']} R={s2['metrics'][m]['recall']}")
        for c in summary["changes"][m]:
            print(" ", c["change"], c["tracklet"], c.get("stage_1", {}).get("successor"), "->",
                  c.get("stage_2", {}).get("successor"), "|", c["explanation"][:150])


if __name__ == "__main__":
    main()

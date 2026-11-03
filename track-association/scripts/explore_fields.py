"""Investigate optional/source-specific fields on the TRAIN partition.

    python -m scripts.explore_fields            # writes results/field_exploration.json

Uses train labels only for the link-level reliability checks (allowed: train is for
developing rules). Never touches val/test labels.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.features import summarize_tracklet
from src.load import load_observations, load_supporting, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "train"


def q(vals, ps=(1, 25, 50, 75, 99)):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    return {f"p{p}": round(float(np.percentile(vals, p)), 3) for p in ps} | {"n": len(vals)}


def main():
    raw = read_jsonl(D / "observations.jsonl")
    tracklets, stats = load_observations(raw)
    sup = {"tracklet_context": read_jsonl(D / "tracklet_context.jsonl"), "id_reports": read_jsonl(D / "id_reports.jsonl")}
    ctx, idr, sstats = load_supporting(sup, tracklets)
    summaries = {t: summarize_tracklet(t, o, ctx.get(t), idr.get(t)) for t, o in tracklets.items()}
    out = {"load_stats": stats, "supporting_stats": sstats}

    # 1. attribute inventory per source
    inv = defaultdict(lambda: defaultdict(list))
    for obs in tracklets.values():
        for o in obs:
            for k, v in o["attributes"].items():
                inv[o["source"]][k].append(v)
    inventory = {}
    for src, fields in sorted(inv.items()):
        n_src = sum(1 for obs in tracklets.values() for o in obs if o["source"] == src)
        inventory[src] = {}
        for k, vals in sorted(fields.items()):
            types = Counter(type(v).__name__ for v in vals)
            entry = {"present_fraction": round(len(vals) / n_src, 3), "types": dict(types),
                     "distinct": len({json.dumps(v) for v in vals})}
            nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if nums:
                entry["quantiles"] = q(nums)
            else:
                entry["top_values"] = Counter(map(str, vals)).most_common(6)
            inventory[src][k] = entry
    out["attribute_inventory"] = inventory

    # 2. speed-like fields vs speed computed from positions (per tracklet medians)
    ratios = defaultdict(list)
    hdg_err = defaultdict(list)
    for tid, obs in tracklets.items():
        s = summaries[tid]
        if not s["end_speed_kmh"] or s["end_speed_kmh"] < 100:
            continue
        src = obs[0]["source"]
        for key in ("gs", "spd"):
            vals = [o["attributes"].get(key) for o in obs[-5:]]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                ratios[f"{src}.{key}"].append(s["end_speed_kmh"] / float(np.median(vals)))
        for key in ("trk", "c2", "hdg"):
            vals = [o["attributes"].get(key) for o in obs[-5:]]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals and s["end_heading"] is not None:
                d = ((vals[-1] - s["end_heading"] + 540) % 360) - 180
                hdg_err[f"{src}.{key}"].append(d)
    out["computed_speed_kmh_over_field"] = {k: q(v) for k, v in ratios.items()}
    out["field_minus_computed_heading_deg"] = {k: q(v) for k, v in hdg_err.items()}

    # 3. feed-b q vs fit residual (does q describe position quality?)
    q_rms = defaultdict(list)
    for tid, obs in tracklets.items():
        if obs[0]["source"] == "feed-b" and summaries[tid]["end_fit_rms_km"] is not None:
            qs = Counter(o["attributes"].get("q") for o in obs).most_common(1)[0][0]
            q_rms[str(qs)].append(summaries[tid]["end_fit_rms_km"])
    out["feedb_q_vs_fit_rms_km"] = {k: q(v) for k, v in sorted(q_rms.items())}
    src_rms = defaultdict(list)
    for tid, obs in tracklets.items():
        if summaries[tid]["end_fit_rms_km"] is not None:
            src_rms[obs[0]["source"]].append(summaries[tid]["end_fit_rms_km"])
    out["fit_rms_km_by_source"] = {k: q(v) for k, v in sorted(src_rms.items())}

    # 4. feed-c alt_ref: altitude offset vs baro? compare alt at link boundaries (train labels)
    labels = {r["tracklet_id"]: r["next_tracklet_id"] for r in read_jsonl(D / "labels.jsonl")}
    alt_pairs = defaultdict(list)
    for a, b in labels.items():
        if not b:
            continue
        sa, sb = summaries[a], summaries[b]
        if sa["end_alt_ft"] is None or sb["start_alt_ft"] is None:
            continue
        ka = tracklets[a][0]["source"] + ("/" + tracklets[a][0]["attributes"].get("alt_ref", "") if tracklets[a][0]["source"] == "feed-c" else "")
        kb = tracklets[b][0]["source"] + ("/" + tracklets[b][0]["attributes"].get("alt_ref", "") if tracklets[b][0]["source"] == "feed-c" else "")
        alt_pairs[f"{ka} -> {kb}"].append(sb["start_alt_ft"] - sa["end_alt_ft"])
    out["true_link_alt_change_ft_by_source_pair"] = {k: q(v) for k, v in sorted(alt_pairs.items()) if len(v) >= 10}

    # 5. identifier / category reliability on true links vs within-tracklet
    def agree(field):
        same = diff = miss = 0
        for a, b in labels.items():
            if not b:
                continue
            va, vb = summaries[a][field], summaries[b][field]
            if va is None or vb is None:
                miss += 1
            elif va == vb:
                same += 1
            else:
                diff += 1
        return {"true_links_same": same, "true_links_different": diff, "true_links_missing": miss}

    out["true_link_agreement"] = {f: agree(f) for f in ("callsign", "cat", "ref")}
    # how often do DIFFERENT aircraft share a value within a scenario? (non-links, random sample)
    sq_within = Counter()
    for tid, obs in tracklets.items():
        if obs[0]["source"] == "feed-a":
            sq_within[len({o["attributes"].get("sq") for o in obs})] += 1
    out["feed-a_sq_distinct_values_per_tracklet"] = dict(sq_within)
    sq_link = {"same": 0, "different": 0}
    for a, b in labels.items():
        if b and tracklets[a][0]["source"] == tracklets[b][0]["source"] == "feed-a":
            sq_link["same" if tracklets[a][-1]["attributes"]["sq"] == tracklets[b][0]["attributes"]["sq"] else "different"] += 1
    out["feed-a_sq_on_true_links"] = sq_link
    out["sq_most_common_values"] = Counter(o["attributes"].get("sq") for obs in tracklets.values() for o in obs if o["source"] == "feed-a").most_common(5)
    out["callsign_conflict_tracklets"] = sum(s["callsign_conflict"] for s in summaries.values())
    out["cat_conflict_tracklets"] = sum(s["cat_conflict"] for s in summaries.values())
    out["receipt_delay_s"] = q([o["late_receipt_s"] for obs in tracklets.values() for o in obs], (50, 90, 99, 99.9))
    out["true_link_gap_min"] = q([(summaries[b]["start_time"] - summaries[a]["end_time"]).total_seconds() / 60 for a, b in labels.items() if b], (50, 90, 99, 100))
    out["altitude_null_fraction_by_source"] = {
        s: round(np.mean([o["alt_ft"] is None for obs in tracklets.values() for o in obs if o["source"] == s]), 3)
        for s in ("feed-a", "feed-b", "feed-c")}
    out["glitch_points_flagged"] = sum(s["n_glitches"] for s in summaries.values())

    path = ROOT / "results" / "field_exploration.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

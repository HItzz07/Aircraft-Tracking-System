"""End-to-end association for one analysis snapshot (used by train, evaluate, demo and API)."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from src.candidates import generate_candidates
from src.decide import check_constraints, decide
from src.features import summarize_tracklet
from src.load import ValidationError, canonical_json, load_observations, load_supporting, read_jsonl
from src.scorers import baseline_score, learned_scores

DEFAULT_SCENARIO = "default"


# ------------------------------------------------------------------ input packaging
def load_folder(folder, scenario=None):
    """Read a supplied data folder into an API-style payload. Labels are NOT included."""
    folder = Path(folder)
    index = {r["tracklet_id"]: r["scenario_id"] for r in read_jsonl(folder / "tracklet_index.jsonl")}
    obs = read_jsonl(folder / "observations.jsonl")
    ctx = read_jsonl(folder / "tracklet_context.jsonl")
    idr = read_jsonl(folder / "id_reports.jsonl")
    if scenario is not None:
        keep = {t for t, s in index.items() if s == scenario}
        obs = [o for o in obs if o.get("tracklet_id") in keep]
        oids = {o["observation_id"] for o in obs}
        ctx = [c for c in ctx if c.get("tracklet_id") in keep]
        idr = [r for r in idr if r.get("observation_id") in oids]
        index = {t: s for t, s in index.items() if t in keep}
    return {"observations": obs, "supporting": {"tracklet_context": ctx, "id_reports": idr},
            "tracklet_scenarios": index}


def sha(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ preparation
def prepare(payload, cand_settings):
    """Validate, dedupe, summarise and generate candidates per scenario."""
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object", code="invalid_body")
    tracklets, stats = load_observations(payload.get("observations"))
    ctx, idr, sstats = load_supporting(payload.get("supporting"), tracklets)
    scen_map = payload.get("tracklet_scenarios") or {}
    if not isinstance(scen_map, dict):
        raise ValidationError("tracklet_scenarios must be an object")
    by_scn = {}
    for tid, obs in tracklets.items():
        sid = scen_map.get(tid, DEFAULT_SCENARIO)
        by_scn.setdefault(sid, []).append(summarize_tracklet(tid, obs, ctx.get(tid), idr.get(tid), sid))
    prepared = []
    for sid in sorted(by_scn):
        summaries = by_scn[sid]
        prepared.append((sid, summaries, generate_candidates(summaries, cand_settings)))
    return tracklets, prepared, {**stats, **sstats}, (ctx, idr)


def score_and_decide(prepared, model, config):
    out = {"baseline": {}, "learned": {}}
    cands_all = []
    for _, summaries, cands in prepared:
        ids = [s["tracklet_id"] for s in summaries]
        b, l = config["baseline"], config["learned"]
        out["baseline"].update(decide(ids, cands, [baseline_score(f, b["variant"]) for _, _, f in cands],
                                      b["threshold"], b["margin"], skip_check=b["skip_check"]))
        out["learned"].update(decide(ids, cands, learned_scores(model, cands, config["features"]),
                                     l["threshold"], l["margin"], skip_check=l["skip_check"]))
        cands_all.extend(cands)
        by_id = {s["tracklet_id"]: s for s in summaries}
        for m in out:
            check_constraints({t: out[m][t] for t in ids}, by_id)
    return out, cands_all


# ------------------------------------------------------------------ serialisation
def _js(v):
    if isinstance(v, datetime):
        return v.isoformat().replace("+00:00", "Z")
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 5)
    return v


def tracklet_view(tid, obs, summary, ctx, idr):
    s = {k: _js(v) for k, v in summary.items() if k not in ("glitch_flags",)}
    return {
        "scenario_id": summary["scenario_id"], "source": summary["source"], "n_points": summary["n_points"],
        "start_time": s["start_time"], "end_time": s["end_time"],
        "points": [{"lat": o["lat"], "lon": o["lon"], "alt_ft": o["alt_ft"], "observed_at": _js(o["observed_at"]),
                    "observation_id": o["observation_id"], "glitch": g}
                   for o, g in zip(obs, summary["glitch_flags"])],
        "raw": [o["raw"] for o in obs],
        "supporting": {"tracklet_context": ctx.get(tid, []), "id_reports_count": len(idr.get(tid, []))},
        "summary": s,
    }


def run_association(payload, model, config, snapshot_id="adhoc", include_tracklets=True):
    tracklets, prepared, stats, (ctx, idr) = prepare(payload, config["candidates"])
    methods, cands = score_and_decide(prepared, model, config)
    # input hash over the canonical, deduplicated input: identical for reordered / duplicated input
    canon_obs = sorted(canonical_json(o["raw"]) for obs in tracklets.values() for o in obs)
    sup = payload.get("supporting") or {}
    canon_sup = {k: sorted({canonical_json(r) for r in (sup.get(k) or [])}) for k in ("tracklet_context", "id_reports")}
    scen_map = payload.get("tracklet_scenarios") or {}
    manifest = {
        "snapshot_id": snapshot_id,
        "input_hash": sha({"obs": canon_obs, "sup": canon_sup, "scn": scen_map}),
        "model_version": config["model_version"],
        "config_hash": sha(config),
        "decision_config": {"candidates": config["candidates"], "baseline": config["baseline"],
                            "learned": config["learned"], "features": config["features"]},
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **stats,
        "n_scenarios": len(prepared),
        "n_candidate_pairs": len(cands),
    }
    resp = {"manifest": manifest, "methods": methods}
    if include_tracklets:
        summ = {s["tracklet_id"]: s for _, ss, _ in prepared for s in ss}
        resp["tracklets"] = {t: tracklet_view(t, tracklets[t], summ[t], ctx, idr) for t in tracklets}
    return resp, cands


def strip_volatile(resp):
    """Result without the generation timestamp, for equality checks."""
    r = json.loads(json.dumps(resp, default=str))
    r["manifest"].pop("generated_at", None)
    return r

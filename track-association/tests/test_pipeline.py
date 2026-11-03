import copy
import json
import random

import joblib

from src.features import PHYSICS_FEATURES
from src.pipeline import prepare, run_association, strip_volatile
from src.scorers import learned_scores


def _decisions(resp):
    return json.dumps(resp["methods"], sort_keys=True)


def test_reorder_and_duplicates_give_identical_results(sample, model, config):
    p = sample[0]
    r1, _ = run_association(copy.deepcopy(p), model, config, "s")
    shuffled = copy.deepcopy(p)
    shuffled["observations"] = shuffled["observations"] + copy.deepcopy(shuffled["observations"][:40])
    random.Random(1).shuffle(shuffled["observations"])
    shuffled["supporting"]["id_reports"] = list(reversed(shuffled["supporting"]["id_reports"])) + shuffled["supporting"]["id_reports"][:5]
    random.Random(2).shuffle(shuffled["supporting"]["tracklet_context"])
    r2, _ = run_association(shuffled, model, config, "s")
    assert _decisions(r1) == _decisions(r2)
    assert r1["manifest"]["input_hash"] == r2["manifest"]["input_hash"]
    assert r2["manifest"]["n_duplicates_removed"] >= 40


def test_model_save_load_consistency(tmp_path, sample, model, config):
    joblib.dump(model, tmp_path / "m.joblib")
    loaded = joblib.load(tmp_path / "m.joblib")
    _, prep, _, _ = prepare(sample[0], config["candidates"])
    cands = [c for _, _, cs in prep for c in cs]
    assert learned_scores(model, cands, config["features"]) == learned_scores(loaded, cands, config["features"])
    r1, _ = run_association(sample[0], model, config, "s")
    r2, _ = run_association(sample[0], loaded, config, "s")
    assert strip_volatile(r1) == strip_volatile(r2)


def test_tracklet_ids_are_not_features(sample, model, config):
    """Renaming every tracklet id must not change any decision (beyond the renaming)."""
    p = copy.deepcopy(sample[0])
    ids = sorted({o["tracklet_id"] for o in p["observations"]})
    new = {t: f"zz-{i:04d}-{t[::-1]}" for i, t in enumerate(reversed(ids))}
    for o in p["observations"]:
        o["tracklet_id"] = new[o["tracklet_id"]]
    for c in p["supporting"]["tracklet_context"]:
        c["tracklet_id"] = new[c["tracklet_id"]]
    p["tracklet_scenarios"] = {new[t]: s for t, s in p["tracklet_scenarios"].items()}
    r1, _ = run_association(sample[0], model, config, "s")
    r2, _ = run_association(p, model, config, "s")
    for m in ("baseline", "learned"):
        a = {t: r["successor"] for t, r in r1["methods"][m].items()}
        b = {t: r2["methods"][m][new[t]]["successor"] for t in a}
        assert b == {t: (new[s] if s else None) for t, s in a.items()}


def test_missing_altitude_handled_end_to_end(sample, model, config):
    p = copy.deepcopy(sample[0])
    for o in p["observations"]:
        o["altitude_ft"] = None
    resp, cands = run_association(p, model, config, "s")
    assert all(c[2]["alt_diff_ft"] != c[2]["alt_diff_ft"] for c in cands)   # NaN everywhere, never 0
    assert all(t["summary"]["end_alt_ft"] is None for t in resp["tracklets"].values())


def test_supporting_streams_optional(sample, model, config):
    p = copy.deepcopy(sample[0])
    p["supporting"] = {}
    resp, _ = run_association(p, model, config, "s")
    assert resp["methods"]["learned"]


def test_physics_only_config_runs(sample, model, config):
    from src.scorers import train_model
    _, prep, _, _ = prepare(sample[0], config["candidates"])
    cands = [c for _, _, cs in prep for c in cs]
    m2 = train_model(cands, sample[1], PHYSICS_FEATURES)
    cfg = dict(config, features=PHYSICS_FEATURES)
    resp, _ = run_association(sample[0], m2, cfg, "s")
    assert resp["manifest"]["decision_config"]["features"] == PHYSICS_FEATURES


def test_scenarios_do_not_mix(sample, model, config):
    p = copy.deepcopy(sample[0])
    ids = sorted(p["tracklet_scenarios"])
    p["tracklet_scenarios"] = {t: f"s{i % 2}" for i, t in enumerate(ids)}
    resp, cands = run_association(p, model, config, "s")
    for a, b, _ in cands:
        assert p["tracklet_scenarios"][a] == p["tracklet_scenarios"][b]

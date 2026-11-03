from datetime import datetime, timedelta, timezone

import pytest

from src.candidates import generate_candidates
from src.decide import check_constraints, decide
from src.features import summarize_tracklet
from src.load import load_observations
from tests.conftest import CAND, straight_tracklet

F = {"gap_min": 5.0, "fwd_err_km": 1.0, "back_err_km": 1.0, "dist_km": 1.0, "heading_change_deg": 0.0}


def test_one_successor_one_predecessor():
    cands = [("A", "B", F), ("A", "C", F), ("D", "B", F)]
    res = decide(["A", "B", "C", "D"], cands, [0.9, 0.2, 0.8], threshold=0.5, margin=0.0)
    succ = [r["successor"] for r in res.values() if r["successor"]]
    assert len(succ) == len(set(succ))
    assert res["A"]["successor"] == "B" and res["B"]["predecessor"] == "A"
    assert res["D"]["successor"] is None and "already accepted" in res["D"]["reason"]


def test_ambiguous_successors_left_unresolved():
    res = decide(["A", "B", "C"], [("A", "B", F), ("A", "C", F)], [0.81, 0.80], threshold=0.5, margin=0.1, skip_check=False)
    assert res["A"]["status"] == "unresolved" and "ambiguous" in res["A"]["reason"]


def test_ambiguous_predecessors_left_unresolved():
    res = decide(["A", "B", "D"], [("A", "B", F), ("D", "B", F)], [0.81, 0.80], threshold=0.5, margin=0.1)
    assert res["A"]["status"] == res["D"]["status"] == "unresolved"


def test_taken_competitor_does_not_block():
    # C is claimed by D (0.95); A's choice between B (0.7) and C (0.69) is then clear
    res = decide(["A", "B", "C", "D"], [("A", "B", F), ("A", "C", F), ("D", "C", F)], [0.7, 0.69, 0.95], 0.5, 0.1)
    assert res["D"]["successor"] == "C" and res["A"]["successor"] == "B"


def test_below_threshold_unresolved():
    res = decide(["A", "B"], [("A", "B", F)], [0.1], threshold=0.5, margin=0)
    assert res["A"]["status"] == "unresolved" and "below threshold" in res["A"]["reason"]


def test_no_candidates_reason():
    res = decide(["A"], [], [], 0.5, 0.0)
    assert res["A"]["status"] == "unresolved" and res["A"]["ranked_candidates"] == []


def test_skip_check_prefers_chain():
    # A->C scores best but B fits between A and C -> accept A->B and B->C, not A->C
    cands = [("A", "C", F), ("A", "B", F), ("B", "C", F)]
    res = decide(["A", "B", "C"], cands, [0.95, 0.9, 0.9], 0.5, 0.1, skip_check=True)
    assert res["A"]["successor"] == "B" and res["B"]["successor"] == "C"
    res2 = decide(["A", "B", "C"], cands, [0.95, 0.9, 0.9], 0.5, 0.0, skip_check=False)
    assert res2["A"]["successor"] == "C"


def test_ranked_candidates_and_measurements():
    res = decide(["A", "B", "C"], [("A", "B", F), ("A", "C", dict(F, alt_diff_ft=float("nan")))], [0.9, 0.3], 0.5, 0.1)
    rc = res["A"]["ranked_candidates"]
    assert [c["candidate"] for c in rc] == ["B", "C"] and rc[0]["rank"] == 1
    assert rc[1]["measurements"]["alt_diff_ft"] is None   # NaN surfaces as null, not 0


def _summaries(records):
    tr, _ = load_observations(records)
    return {t: summarize_tracklet(t, o) for t, o in tr.items()}


def test_candidates_require_strict_time_order():
    t0 = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    a = straight_tracklet("A", t0, 20, 50.0, 0.0, 0.0005)
    b = straight_tracklet("B", t0 + timedelta(seconds=190), 20, 50.0095, 0.0, 0.0005)   # starts exactly when A ends
    c = straight_tracklet("C", t0 + timedelta(seconds=100), 20, 50.005, 0.0, 0.0005)    # overlaps A
    d = straight_tracklet("D", t0 + timedelta(seconds=400), 20, 50.02, 0.0, 0.0005)
    s = _summaries(a + b + c + d)
    pairs = {(x, y) for x, y, _ in generate_candidates(list(s.values()), CAND)}
    assert ("A", "B") not in pairs and ("A", "C") not in pairs and ("A", "D") in pairs
    for x, y in pairs:
        assert s[y]["start_time"] > s[x]["end_time"]


def test_check_constraints_detects_violations():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    summ = {k: {"start_time": t + timedelta(minutes=i * 10), "end_time": t + timedelta(minutes=i * 10 + 5)} for i, k in enumerate("ABC")}
    ok = {"A": {"successor": "B"}, "B": {"successor": "C"}, "C": {"successor": None}}
    check_constraints(ok, summ)
    with pytest.raises(AssertionError):
        check_constraints({"A": {"successor": "C"}, "B": {"successor": "C"}, "C": {"successor": None}}, summ)
    with pytest.raises(AssertionError):
        check_constraints({"A": {"successor": None}, "B": {"successor": None}, "C": {"successor": "A"}}, summ)


def test_real_scenario_respects_constraints(sample, model, config):
    from src.pipeline import prepare, score_and_decide
    _, prep, _, _ = prepare(sample[0], config["candidates"])
    out, _ = score_and_decide(prep, model, config)   # calls check_constraints internally
    for m in out.values():
        preds = [r["successor"] for r in m.values() if r["successor"]]
        assert len(preds) == len(set(preds))

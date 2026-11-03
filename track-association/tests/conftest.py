import copy
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_data as gen  # noqa: E402
from src.features import ALL_FEATURES  # noqa: E402
from src.pipeline import prepare  # noqa: E402
from src.scorers import train_model  # noqa: E402

CAND = {"max_gap_min": 60.0, "max_implied_speed_kmh": 1110.0, "max_err_km_per_min": 11.7, "err_offset_min": 2.0}


def rec(obs_id="o1", tid="t1", t="2026-01-10T10:00:00Z", lat=10.0, lon=20.0, alt=None, src="feed-a", rx=None, attrs=None):
    return {"observation_id": obs_id, "tracklet_id": tid, "observed_at": t,
            "received_at": rx or "2026-01-10T10:05:00Z", "latitude": lat, "longitude": lon,
            "altitude_ft": alt, "source": src, "attributes": attrs if attrs is not None else {"x": [1, 2]}}


def straight_tracklet(tid, t0, n, lat0, lon0, dlat_per_step, step_s=10, alt=30000):
    out = []
    for i in range(n):
        t = t0 + timedelta(seconds=i * step_s)
        out.append(rec(f"{tid}-o{i:03d}", tid, t.strftime("%Y-%m-%dT%H:%M:%SZ"), lat0 + i * dlat_per_step, lon0, alt))
    return out


def scenario_payload(seed):
    """A realistic generated scenario (labels from generator truth)."""
    rng = random.Random(f"test-{seed}")
    obs, idr, ctx, truth = gen.gen_scenario(rng, f"scn-test-{seed}", datetime(2026, 9, 1, 8, tzinfo=timezone.utc))
    labels = {r["tracklet_id"]: r["next_tracklet_id"] for r in gen.labels_from_truth(truth)}
    payload = {"observations": obs, "supporting": {"tracklet_context": ctx, "id_reports": idr},
               "tracklet_scenarios": {t["tracklet_id"]: t["scenario_id"] for t in truth}}
    return payload, labels


@pytest.fixture(scope="session")
def config():
    return {"model_version": "test-0", "features": ALL_FEATURES, "feature_set": "physics+supporting",
            "candidates": CAND,
            "baseline": {"variant": "per_minute", "threshold": -3.0, "margin": 0.1, "skip_check": True},
            "learned": {"threshold": 0.3, "margin": 0.05, "skip_check": True}}


@pytest.fixture(scope="session")
def model():
    cands, labels = [], {}
    for s in range(3):
        p, lab = scenario_payload(s)
        _, prep, _, _ = prepare(p, CAND)
        cands += [c for _, _, cs in prep for c in cs]
        labels.update(lab)
    return train_model(cands, labels, ALL_FEATURES)


@pytest.fixture(scope="session")
def sample():
    return scenario_payload(99)


@pytest.fixture
def sample_copy(sample):
    return copy.deepcopy(sample[0])

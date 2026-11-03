# Aircraft Track Association — Implementation Guide

A step-by-step plan with simple code for the Govia ML challenge.
Goal: join broken pieces of aircraft paths ("tracklets") back together, carefully, and explain every decision.

> **Before you start:** read the data package (source notes, schema, label format, snapshot files).
> Anything in this guide marked **ASSUMPTION** must be checked against the real data and adjusted.

---

## Table of contents

1. [The problem in plain words](#1-the-problem-in-plain-words)
2. [Project structure](#2-project-structure)
3. [Setup](#3-setup)
4. [Step 1 — Load, validate, deduplicate](#4-step-1--load-validate-deduplicate)
5. [Step 2 — Explore the optional fields](#5-step-2--explore-the-optional-fields)
6. [Step 3 — Summarize each tracklet](#6-step-3--summarize-each-tracklet)
7. [Step 4 — Generate candidate pairs](#7-step-4--generate-candidate-pairs)
8. [Step 5 — Baseline scorer](#8-step-5--baseline-scorer)
9. [Step 6 — Learned scorer](#9-step-6--learned-scorer)
10. [Step 7 — Decide links (greedy)](#10-step-7--decide-links-greedy)
11. [Step 8 — Evaluate](#11-step-8--evaluate)
12. [Step 9 — Train command](#12-step-9--train-command)
13. [Step 10 — Evaluate command and stress test](#13-step-10--evaluate-command-and-stress-test)
14. [Step 11 — Staged evidence demo](#14-step-11--staged-evidence-demo)
15. [Step 12 — Backend API](#15-step-12--backend-api)
16. [Step 13 — Web interface](#16-step-13--web-interface)
17. [Step 14 — Tests](#17-step-14--tests)
18. [Step 15 — README and submission](#18-step-15--readme-and-submission)
19. [Time plan](#19-time-plan)
20. [Final checklist](#20-final-checklist)

---

## 1. The problem in plain words

- Planes fly. Their recorded paths break into short pieces called **tracklets**.
- Each tracklet belongs to exactly one plane, but we don't know which plane.
- We must say: **"tracklet B is the next piece after tracklet A, same plane"** → link `A → B`.
- If we are not sure, we say **unresolved**. A wrong link is worse than no link.

```
Plane X:   A ━━━━━━      gap      B ━━━━━━      gap      C ━━━━━
Plane Y:        D ━━━━━━━━━          E ━━━━━
Answer:    A → B,  B → C,  D → E
```

### Rules for a link A → B

| Rule | Meaning |
|---|---|
| Time order | B starts **strictly after** A ends |
| One-to-one | Each tracklet: max 1 successor, max 1 predecessor |
| No cycles | Guaranteed automatically by the time-order rule |
| Immediate | If A → B → C is true, then A → C is **wrong** |
| Same scenario | Never link across scenarios |
| Gap ≠ evidence | A time gap alone does not prove a link |

### The whole pipeline

```
raw JSONL ─► validate & dedupe ─► tracklet summaries ─► candidate pairs
                                                          │
                                  ┌───────────────────────┴───────────────┐
                                  ▼                                       ▼
                          baseline score (physics)              learned score (classifier)
                                  └───────────────────────┬───────────────┘
                                                          ▼
                                       greedy decision (threshold + margin + 1-to-1)
                                                          ▼
                                       results JSON + metrics + API + UI
```

---

## 2. Project structure

```
track-association/
├── data/                      # supplied data package (do not modify)
├── src/
│   ├── load.py                # validation, dedupe, grouping
│   ├── geo.py                 # distance, bearing, projection
│   ├── features.py            # tracklet summaries + pair features
│   ├── candidates.py          # candidate generation
│   ├── scorers.py             # baseline + learned scoring
│   ├── decide.py              # greedy association
│   ├── evaluate.py            # metrics
│   └── pipeline.py            # run everything for one snapshot
├── scripts/
│   ├── explore_fields.py      # field investigation
│   ├── train.py               # python -m scripts.train
│   ├── evaluate.py            # python -m scripts.evaluate
│   └── staged_demo.py         # python -m scripts.staged_demo
├── api/
│   ├── main.py                # FastAPI app
│   └── static/index.html      # web UI
├── artifacts/                 # model.joblib, config.json (created by train)
├── results/                   # evaluation + staged demo outputs
├── tests/
├── requirements.txt
├── README.md
└── PROMPTS.md                 # AI prompt log (required)
```

---

## 3. Setup

`requirements.txt`

```
fastapi==0.115.*
uvicorn==0.30.*
pydantic==2.*
scikit-learn==1.5.*
numpy==1.26.*
pandas==2.2.*
joblib==1.4.*
pytest==8.*
httpx==0.27.*
```

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All randomness uses one seed: `SEED = 42`.

---

## 4. Step 1 — Load, validate, deduplicate

**What it does**

1. Checks required fields, non-empty IDs, UTC timestamps, valid lat/lon.
2. Same `observation_id` + same content → keep one copy (duplicate).
3. Same `observation_id` + different content → **error naming the ID**.
4. Groups by tracklet and sorts by `observed_at` (never by `received_at`).
5. Keeps the raw record untouched for the UI.

`src/load.py`

```python
import json
from datetime import datetime
from collections import defaultdict

REQUIRED = ["observation_id", "tracklet_id", "observed_at",
            "received_at", "latitude", "longitude", "source"]


class ValidationError(Exception):
    def __init__(self, message, observation_id=None):
        super().__init__(message)
        self.observation_id = observation_id


def read_jsonl(path):
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValidationError(f"{path}:{line_no} is not valid JSON")
    return records


def parse_utc(value, field, obs_id):
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string", obs_id)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError(f"{field} is not a valid timestamp: {value}", obs_id)
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValidationError(f"{field} must be UTC: {value}", obs_id)
    return dt


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(rec):
    """Check one raw record. Returns a clean copy; never edits the raw record."""
    if not isinstance(rec, dict):
        raise ValidationError("record must be a JSON object")
    obs_id = rec.get("observation_id")

    for field in REQUIRED:
        if field not in rec or rec[field] is None:
            raise ValidationError(f"missing required field: {field}", obs_id)

    for field in ["observation_id", "tracklet_id", "source"]:
        if not isinstance(rec[field], str) or not rec[field].strip():
            raise ValidationError(f"{field} must be a non-empty string", obs_id)

    lat, lon = rec["latitude"], rec["longitude"]
    if not is_number(lat) or not -90 <= lat <= 90:
        raise ValidationError(f"invalid latitude: {lat}", obs_id)
    if not is_number(lon) or not -180 <= lon <= 180:
        raise ValidationError(f"invalid longitude: {lon}", obs_id)

    alt = rec.get("altitude_ft")
    if alt is not None and not is_number(alt):
        raise ValidationError(f"altitude_ft must be a number or null", obs_id)

    attrs = rec.get("attributes")
    if attrs is not None and not isinstance(attrs, dict):
        raise ValidationError("attributes must be an object", obs_id)

    return {
        "observation_id": obs_id,
        "tracklet_id": rec["tracklet_id"],
        "observed_at": parse_utc(rec["observed_at"], "observed_at", obs_id),
        "received_at": parse_utc(rec["received_at"], "received_at", obs_id),
        "lat": float(lat),
        "lon": float(lon),
        "alt_ft": None if alt is None else float(alt),   # null stays null, never 0
        "source": rec["source"],
        "attributes": attrs or {},
        "raw": rec,                                      # original, for inspection
    }


def dedupe(raw_records):
    """
    Same id + same content  -> keep one (duplicate).
    Same id + different content -> error.
    Python dict == ignores key order, keeps missing != null, keeps list order.
    """
    seen = {}
    for rec in raw_records:
        obs_id = rec.get("observation_id")
        if obs_id in seen:
            if seen[obs_id] != rec:
                raise ValidationError(
                    f"conflicting content for observation_id {obs_id}", obs_id)
            continue
        seen[obs_id] = rec
    return list(seen.values())


def load_observations(raw_records):
    """Validate + dedupe + group into tracklets sorted by observed_at."""
    unique = dedupe(raw_records)
    clean = [validate(r) for r in unique]

    tracklets = defaultdict(list)
    for obs in clean:
        tracklets[obs["tracklet_id"]].append(obs)

    for tid in tracklets:
        # observation_id is only a tie-breaker so ordering is deterministic
        tracklets[tid].sort(key=lambda o: (o["observed_at"], o["observation_id"]))
    return dict(tracklets)
```

> **ASSUMPTION:** how a tracklet maps to a scenario. Check the data package
> (separate file per scenario? a `scenario_id` in tracklet metadata?) and write
> one small function `scenario_of(tracklet_id)` for it. Use it **only** for grouping.

---

## 5. Step 2 — Explore the optional fields

The brief grades **how you investigate**, not how many fields you use.
Run a script, look at the numbers, and write down what you find.

`scripts/explore_fields.py`

```python
import sys
import pandas as pd
from src.load import read_jsonl, load_observations
from src.features import summarize_tracklet

records = read_jsonl(sys.argv[1])
tracklets = load_observations(records)

rows = []
for tid, obs_list in tracklets.items():
    for o in obs_list:
        row = {"tracklet_id": tid, "source": o["source"], "alt_ft": o["alt_ft"]}
        for key, value in o["attributes"].items():
            row[f"attr.{key}"] = value
        rows.append(row)
df = pd.DataFrame(rows)

# 1. Which attributes exist per source? (same name may mean different things!)
for source, group in df.groupby("source"):
    print(f"\n=== {source}: {len(group)} rows ===")
    cols = [c for c in group.columns if c.startswith("attr.")]
    print(group[cols].describe(include="all").T[["count", "unique", "mean", "min", "max"]])

# 2. Does a "speed-like" field match speed computed from positions?
#    Compare per tracklet: median(attr) vs computed speed in knots.
```

Fill a table like this in the README:

| Source | Field | Evidence | Interpretation | Status | Used? |
|---|---|---|---|---|---|
| feed-b | `spd` | range ~150–500, correlates with computed speed | speed, likely knots | inferred | maybe |
| feed-b | `c2` | range 0–360, matches computed bearing | heading in degrees | inferred | maybe |
| feed-b | `ref` | 6 hex chars, sometimes changes inside a tracklet | reported identifier, unreliable | inferred | no / test |
| feed-b | `q` | small set of values | quality flag, meaning unknown | unknown | no |
| feed-b | `node` | ingest server names | infrastructure, irrelevant | inferred | no |

The example values above are **guesses**. Replace them with what the data shows.
It's fine to use only time and position. The task is solvable without optional fields.

---

## 6. Step 3 — Summarize each tracklet

For every tracklet we need: where and when it **starts**, where and when it **ends**,
and how it was **moving at the end** (speed + heading).

`src/geo.py`

```python
import math

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Direction from point 1 to point 2. 0 = north, 90 = east."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a, b):
    """Smallest difference between two headings, 0..180."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def project(lat, lon, speed_kmh, heading, hours):
    """Move a point forward in a straight line (simple flat approximation)."""
    dist = speed_kmh * hours
    dlat = dist * math.cos(math.radians(heading)) / EARTH_R_KM
    dlon = dist * math.sin(math.radians(heading)) / (
        EARTH_R_KM * max(math.cos(math.radians(lat)), 1e-6))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)
```

`src/features.py` (part 1)

```python
from src.geo import haversine_km, bearing_deg

MIN_VELOCITY_WINDOW_S = 60   # use points at least 60 s apart to reduce noise


def end_velocity(obs):
    """Speed (km/h) and heading at the END of a tracklet, from positions only."""
    if len(obs) < 2:
        return None, None
    last = obs[-1]
    # walk backwards until we find a point far enough back in time
    for prev in reversed(obs[:-1]):
        dt_h = (last["observed_at"] - prev["observed_at"]).total_seconds() / 3600
        if dt_h * 3600 >= MIN_VELOCITY_WINDOW_S:
            dist = haversine_km(prev["lat"], prev["lon"], last["lat"], last["lon"])
            return dist / dt_h, bearing_deg(prev["lat"], prev["lon"], last["lat"], last["lon"])
    return None, None


def start_velocity(obs):
    """Same idea at the START of a tracklet (reverse the list)."""
    if len(obs) < 2:
        return None, None
    first = obs[0]
    for nxt in obs[1:]:
        dt_h = (nxt["observed_at"] - first["observed_at"]).total_seconds() / 3600
        if dt_h * 3600 >= MIN_VELOCITY_WINDOW_S:
            dist = haversine_km(first["lat"], first["lon"], nxt["lat"], nxt["lon"])
            return dist / dt_h, bearing_deg(first["lat"], first["lon"], nxt["lat"], nxt["lon"])
    return None, None


def last_known_alt(obs, from_end=True):
    seq = reversed(obs) if from_end else obs
    for o in seq:
        if o["alt_ft"] is not None:
            return o["alt_ft"]
    return None   # explicitly missing


def summarize_tracklet(tid, obs):
    end_speed, end_heading = end_velocity(obs)
    start_speed, start_heading = start_velocity(obs)
    return {
        "tracklet_id": tid,
        "n_points": len(obs),
        "start_time": obs[0]["observed_at"],
        "end_time": obs[-1]["observed_at"],
        "start_lat": obs[0]["lat"], "start_lon": obs[0]["lon"],
        "end_lat": obs[-1]["lat"], "end_lon": obs[-1]["lon"],
        "end_speed_kmh": end_speed, "end_heading": end_heading,
        "start_speed_kmh": start_speed, "start_heading": start_heading,
        "end_alt_ft": last_known_alt(obs, from_end=True),
        "start_alt_ft": last_known_alt(obs, from_end=False),
    }
```

**Irregular sampling** is handled because we use real time differences, never "number of points".
**Missing altitude** stays `None`, so the altitude feature becomes "missing", not 0.

---

## 7. Step 4 — Generate candidate pairs

Cheap filter: keep only pairs that are *physically possible*.
Both scorers use the **same** candidate set, so the comparison is fair.

`src/features.py` (part 2: pair features)

```python
import math
from src.geo import haversine_km, bearing_deg, angle_diff, project

FEATURES = ["gap_min", "dist_km", "implied_speed_kmh", "pred_error_km",
            "speed_diff_kmh", "heading_change", "alt_diff_ft"]


def pair_features(a, b):
    gap_h = (b["start_time"] - a["end_time"]).total_seconds() / 3600
    dist = haversine_km(a["end_lat"], a["end_lon"], b["start_lat"], b["start_lon"])
    implied_speed = dist / gap_h

    # Where would A be at B's start time if it kept flying straight?
    if a["end_speed_kmh"] is not None:
        plat, plon = project(a["end_lat"], a["end_lon"],
                             a["end_speed_kmh"], a["end_heading"], gap_h)
        pred_error = haversine_km(plat, plon, b["start_lat"], b["start_lon"])
    else:
        pred_error = dist          # no velocity known: fall back to raw distance

    speed_diff = (abs(a["end_speed_kmh"] - b["start_speed_kmh"])
                  if a["end_speed_kmh"] is not None and b["start_speed_kmh"] is not None
                  else math.nan)

    heading_change = (angle_diff(a["end_heading"], b["start_heading"])
                      if a["end_heading"] is not None and b["start_heading"] is not None
                      else math.nan)

    alt_diff = (abs(a["end_alt_ft"] - b["start_alt_ft"])
                if a["end_alt_ft"] is not None and b["start_alt_ft"] is not None
                else math.nan)     # NaN = "unknown", NOT zero

    return {
        "gap_min": gap_h * 60,
        "dist_km": dist,
        "implied_speed_kmh": implied_speed,
        "pred_error_km": pred_error,
        "speed_diff_kmh": speed_diff,
        "heading_change": heading_change,
        "alt_diff_ft": alt_diff,
    }
```

`src/candidates.py`

```python
from src.features import pair_features


def generate_candidates(summaries, settings):
    """
    summaries: list of tracklet summaries from ONE scenario.
    settings:  {"max_gap_min": ..., "max_implied_speed_kmh": ...}
    Returns list of (a_id, b_id, features).
    """
    candidates = []
    for a in summaries:
        for b in summaries:
            if a["tracklet_id"] == b["tracklet_id"]:
                continue
            if b["start_time"] <= a["end_time"]:          # strictly after
                continue
            f = pair_features(a, b)
            if f["gap_min"] > settings["max_gap_min"]:
                continue
            if f["implied_speed_kmh"] > settings["max_implied_speed_kmh"]:
                continue
            candidates.append((a["tracklet_id"], b["tracklet_id"], f))
    return candidates
```

**How to pick the settings:** look at true links in **train** (gap and implied speed),
pick limits that keep ~99% of them, confirm on **validation**, then freeze.
Report **candidate recall** = true links kept ÷ all true links.

> Use the speed limit from the data package's "simplified movement assumptions" if one is given.

---

## 8. Step 5 — Baseline scorer

A simple physics rule. **Score direction: higher = better** for both scorers,
so the decision code is the same.

`src/scorers.py` (part 1)

```python
def baseline_score(f):
    """
    Score = minus the prediction error in km.
    0 means perfect continuation; -20 means B starts 20 km from where A should be.
    Not a probability.
    """
    return -f["pred_error_km"]
```

Why this is enough as a baseline: it's one number, easy to explain,
and it uses only documented fields (time + position).

---

## 9. Step 6 — Learned scorer

Turn every candidate pair in **train** into a row: features → label (1 = true next piece, 0 = not).
`HistGradientBoostingClassifier` handles `NaN` (missing values) on its own, so no fake zeros.

`src/scorers.py` (part 2)

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import FEATURES

SEED = 42


def to_matrix(candidates):
    return np.array([[f[name] for name in FEATURES] for _, _, f in candidates], dtype=float)


def train_model(candidates, true_next):
    """
    candidates: list of (a_id, b_id, features) from TRAIN
    true_next:  dict a_id -> correct b_id (or None)
    """
    X = to_matrix(candidates)
    y = np.array([1 if true_next.get(a) == b else 0 for a, b, _ in candidates])
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                           random_state=SEED)
    model.fit(X, y)
    return model


def learned_scores(model, candidates):
    """Score = model output between 0 and 1, higher = better. Not calibrated."""
    if not candidates:
        return []
    return model.predict_proba(to_matrix(candidates))[:, 1].tolist()
```

Labels are used **only** here (training) and in evaluation. Never at inference.

---

## 10. Step 7 — Decide links (greedy)

Same function for both scorers. Only the threshold and margin differ.

**Rules, in order:**

1. Score below threshold → reject (*"too weak"*).
2. A already has a successor, or B already has a predecessor → reject (*"conflict"*).
3. The next-best option is too close (for A's successors **or** B's predecessors) → reject (*"ambiguous"*).
4. Otherwise → accept.

We go from the best pair to the worst, so stronger links win conflicts.

`src/decide.py`

```python
from collections import defaultdict

TOP_K = 5


def second_best(scores_list, exclude_score):
    rest = sorted(scores_list, reverse=True)
    rest.remove(exclude_score)
    return rest[0] if rest else None


def decide(tracklet_ids, candidates, scores, threshold, margin):
    """
    candidates: list of (a_id, b_id, features)
    scores:     list of floats, same order, higher = better
    Returns {tracklet_id: result_dict}
    """
    pairs = [(a, b, f, s) for (a, b, f), s in zip(candidates, scores)]

    scores_from_a = defaultdict(list)    # all successor scores for A
    scores_into_b = defaultdict(list)    # all predecessor scores for B
    for a, b, _, s in pairs:
        scores_from_a[a].append(s)
        scores_into_b[b].append(s)

    # best first; IDs only break ties so results are deterministic
    pairs.sort(key=lambda p: (-p[3], p[0], p[1]))

    successor, predecessor, reasons = {}, {}, {}

    for a, b, f, s in pairs:
        if a in successor:
            continue                      # A already decided
        if s < threshold:
            reasons.setdefault(a, f"best candidate {b} below threshold "
                                  f"({s:.3f} < {threshold})")
            continue
        if b in predecessor:
            reasons.setdefault(a, f"{b} already taken by stronger link "
                                  f"from {predecessor[b]}")
            continue
        alt_a = second_best(scores_from_a[a], s)
        alt_b = second_best(scores_into_b[b], s)
        if alt_a is not None and s - alt_a < margin:
            reasons.setdefault(a, f"ambiguous: {b} only {s - alt_a:.3f} ahead of "
                                  f"another successor")
            continue
        if alt_b is not None and s - alt_b < margin:
            reasons.setdefault(a, f"ambiguous: another tracklet fits into {b} "
                                  f"almost as well")
            continue

        successor[a], predecessor[b] = b, a
        reasons[a] = (f"accepted {b}: gap {f['gap_min']:.1f} min, "
                      f"prediction error {f['pred_error_km']:.2f} km, score {s:.3f}")

    # Build output for EVERY tracklet
    ranked = defaultdict(list)
    for a, b, f, s in sorted(pairs, key=lambda p: (-p[3], p[1])):
        ranked[a].append({"candidate": b, "score": round(s, 4),
                          "measurements": {k: round(v, 3) for k, v in f.items()}})

    results = {}
    for tid in sorted(tracklet_ids):
        accepted = successor.get(tid)
        results[tid] = {
            "status": "accepted" if accepted else "unresolved",
            "successor": accepted,
            "predecessor": predecessor.get(tid),
            "ranked_candidates": ranked[tid][:TOP_K],
            "reason": reasons.get(tid, "no plausible candidate"),
        }
    return results
```

**Constraint guarantees (say these in the review):**

- One successor: we skip A once it's in `successor`.
- One predecessor: we skip B once it's in `predecessor`.
- No cycles: every link goes forward in time, so a loop is impossible.

**Known limitation:** greedy is not globally optimal. One strong wrong link can block a correct one.
The margin check reduces this risk. That's acceptable per the brief.

---

## 11. Step 8 — Evaluate

> **ASSUMPTION:** labels look like `{"tracklet_id": "t1", "next_tracklet_id": "t2" or null}`.
> Adjust `load_labels` to the real format.

`src/evaluate.py`

```python
import json


def load_labels(path):
    labels = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                labels[row["tracklet_id"]] = row.get("next_tracklet_id")
    return labels


def safe_div(n, d):
    return None if d == 0 else n / d      # None = "N/A", never fake 100%


def metrics(results, labels, candidates):
    accepted = {a: r["successor"] for a, r in results.items() if r["successor"]}
    true_links = {a: b for a, b in labels.items() if b is not None}
    candidate_pairs = {(a, b) for a, b, _ in candidates}

    correct = sum(1 for a, b in accepted.items() if labels.get(a) == b)
    incorrect = len(accepted) - correct
    unresolved = sum(1 for a in true_links if a not in accepted)
    kept = sum(1 for a, b in true_links.items() if (a, b) in candidate_pairs)

    return {
        "accepted_links": len(accepted),
        "correct_links": correct,
        "incorrect_links": incorrect,
        "labeled_links": len(true_links),
        "precision": safe_div(correct, len(accepted)),
        "recall": safe_div(correct, len(true_links)),
        "unresolved_proportion": safe_div(unresolved, len(true_links)),
        "candidate_recall": safe_div(kept, len(true_links)),
    }
```

Evaluation runs over **every tracklet in the partition**. Links dropped by candidate
generation simply count as missed. We don't evaluate only on sampled pairs.

---

## 12. Step 9 — Train command

`src/pipeline.py`: one function used by train, evaluate, API and demo.

```python
from src.load import load_observations
from src.features import summarize_tracklet
from src.candidates import generate_candidates
from src.scorers import baseline_score, learned_scores
from src.decide import decide


def prepare(raw_records, scenario_of, settings):
    """Returns per-scenario summaries and candidates."""
    tracklets = load_observations(raw_records)
    by_scenario = {}
    for tid, obs in tracklets.items():
        by_scenario.setdefault(scenario_of(tid), []).append(summarize_tracklet(tid, obs))

    prepared = []
    for scenario, summaries in sorted(by_scenario.items()):
        cands = generate_candidates(summaries, settings)
        prepared.append((scenario, summaries, cands))
    return tracklets, prepared


def run_association(raw_records, scenario_of, model, config):
    tracklets, prepared = prepare(raw_records, scenario_of, config["candidates"])
    out = {"baseline": {}, "learned": {}, "candidates": []}
    for scenario, summaries, cands in prepared:
        ids = [s["tracklet_id"] for s in summaries]
        b = config["baseline"]
        l = config["learned"]
        out["baseline"].update(decide(ids, cands, [baseline_score(f) for _, _, f in cands],
                                      b["threshold"], b["margin"]))
        out["learned"].update(decide(ids, cands, learned_scores(model, cands),
                                     l["threshold"], l["margin"]))
        out["candidates"].extend(cands)
    return tracklets, out
```

`scripts/train.py`: fit on **train**, choose settings on **validation**, save everything.

```python
import json, joblib, itertools
from src.load import read_jsonl
from src.evaluate import load_labels, metrics
from src.pipeline import prepare
from src.scorers import train_model, baseline_score, learned_scores
from src.decide import decide

MODEL_VERSION = "v1.0.0"
CAND = {"max_gap_min": 60, "max_implied_speed_kmh": 1200}   # tune from train data
MIN_PRECISION = 0.95       # wrong links are costly: protect precision first

# --- 1. Fit on train ---
train_raw = read_jsonl("data/train/observations.jsonl")       # ASSUMPTION: paths
train_labels = load_labels("data/train/labels.jsonl")
_, train_prep = prepare(train_raw, scenario_of, CAND)
train_cands = [c for _, _, cands in train_prep for c in cands]
model = train_model(train_cands, train_labels)

# --- 2. Sweep thresholds on validation ---
val_raw = read_jsonl("data/val/observations.jsonl")
val_labels = load_labels("data/val/labels.jsonl")
_, val_prep = prepare(val_raw, scenario_of, CAND)


def evaluate_setting(score_fn, threshold, margin):
    results, all_cands = {}, []
    for _, summaries, cands in val_prep:
        ids = [s["tracklet_id"] for s in summaries]
        results.update(decide(ids, cands, score_fn(cands), threshold, margin))
        all_cands += cands
    return metrics(results, val_labels, all_cands)


def pick(score_fn, thresholds, margins):
    """Highest recall among settings with precision >= MIN_PRECISION."""
    table, best = [], None
    for t, m in itertools.product(thresholds, margins):
        r = evaluate_setting(score_fn, t, m)
        table.append({"threshold": t, "margin": m, **r})
        ok = r["precision"] is not None and r["precision"] >= MIN_PRECISION
        if ok and (best is None or r["recall"] > best["recall"]):
            best = {"threshold": t, "margin": m, **r}
    return best, table


base_fn = lambda cands: [baseline_score(f) for _, _, f in cands]
learn_fn = lambda cands: learned_scores(model, cands)

base_best, base_table = pick(base_fn, [-1, -2, -5, -10, -20], [0, 1, 2, 5])
learn_best, learn_table = pick(learn_fn, [0.3, 0.5, 0.7, 0.9], [0, 0.05, 0.1, 0.2])

# --- 3. Save artifacts ---
config = {
    "model_version": MODEL_VERSION,
    "seed": 42,
    "candidates": CAND,
    "baseline": {"threshold": base_best["threshold"], "margin": base_best["margin"]},
    "learned": {"threshold": learn_best["threshold"], "margin": learn_best["margin"]},
    "selection_rule": f"max recall with validation precision >= {MIN_PRECISION}",
}
joblib.dump(model, "artifacts/model.joblib")
json.dump(config, open("artifacts/config.json", "w"), indent=2)
json.dump({"baseline": base_table, "learned": learn_table},
          open("results/validation_sweep.json", "w"), indent=2, default=str)
print(json.dumps(config, indent=2))
```

```bash
python -m scripts.train
```

The `validation_sweep.json` table is your **"conservative vs permissive"** comparison.

---

## 13. Step 10 — Evaluate command and stress test

`scripts/evaluate.py`

```python
import json, joblib, random
from src.load import read_jsonl
from src.evaluate import load_labels, metrics
from src.pipeline import run_association

model = joblib.load("artifacts/model.joblib")
config = json.load(open("artifacts/config.json"))

test_raw = read_jsonl("data/test/observations.jsonl")
test_labels = load_labels("data/test/labels.jsonl")


def report(raw, name):
    _, out = run_association(raw, scenario_of, model, config)
    return {
        "dataset": name,
        "model_version": config["model_version"],
        "baseline": metrics(out["baseline"], test_labels, out["candidates"]),
        "learned": metrics(out["learned"], test_labels, out["candidates"]),
    }


def thin(raw, keep_every=2):
    """Stress test: keep every 2nd point per tracklet, always keep first + last."""
    by_t = {}
    for r in raw:
        by_t.setdefault(r["tracklet_id"], []).append(r)
    kept = []
    for recs in by_t.values():
        recs = sorted(recs, key=lambda r: r["observed_at"])
        for i, r in enumerate(recs):
            if i % keep_every == 0 or i == len(recs) - 1:
                kept.append(r)
    return kept


results = [report(test_raw, "test"), report(thin(test_raw), "test_thinned_50pct")]
json.dump(results, open("results/test_metrics.json", "w"), indent=2)
print(json.dumps(results, indent=2))
```

```bash
python -m scripts.evaluate
```

- Run on test **once**, after freezing. If you change anything afterwards, write that down.
- Stress test: same labels, same model, **no retraining**.
- After that, inspect **at least 3 errors** (wrong link, missed link, ambiguous case)
  and **one case where baseline and learned disagree**. Write them up in `results/failure_analysis.md`.

---

## 14. Step 11 — Staged evidence demo

Snapshot 1 and snapshot 2 are each **complete** inputs. Run them separately with the
**same** model and config. Never compute snapshot 1 features from snapshot 2 data.

`scripts/staged_demo.py`

```python
import json, joblib
from src.load import read_jsonl
from src.pipeline import run_association

model = joblib.load("artifacts/model.joblib")
config = json.load(open("artifacts/config.json"))


def run(path, snapshot_id):
    raw = read_jsonl(path)
    _, out = run_association(raw, scenario_of, model, config)
    result = {"snapshot_id": snapshot_id, "input_file": path,
              "model_version": config["model_version"], "config": config,
              "baseline": out["baseline"], "learned": out["learned"]}
    json.dump(result, open(f"results/{snapshot_id}.json", "w"), indent=2, default=str)
    return result


def diff(r1, r2, method):
    changes = []
    for tid in sorted(set(r1[method]) | set(r2[method])):
        before = r1[method].get(tid, {}).get("successor")
        after = r2[method].get(tid, {}).get("successor")
        if before != after:
            kind = ("new link" if before is None else
                    "withdrawn" if after is None else "replaced")
            changes.append({"tracklet": tid, "before": before, "after": after,
                            "change": kind,
                            "reason_after": r2[method].get(tid, {}).get("reason")})
    return changes


s1 = run("data/staged/snapshot_1.jsonl", "stage_1")     # ASSUMPTION: paths
s2 = run("data/staged/snapshot_2.jsonl", "stage_2")
summary = {m: diff(s1, s2, m) for m in ["baseline", "learned"]}
json.dump(summary, open("results/staged_diff.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
```

```bash
python -m scripts.staged_demo
```

**How to explain a "replaced" link:**

> In stage 1 we accepted A → C. In stage 2 a delayed tracklet B arrived, and it fits inside
> the A–C gap (smaller prediction error both ways). Now we accept A → B and B → C.
> The aircraft identity is the same. We only found the missing middle piece.

If nothing changes, don't force it. Explain why the new evidence didn't justify a change.

---

## 15. Step 12 — Backend API

**Request** `POST /tracklets/associate`

```json
{
  "snapshot_id": "stage_1",
  "records": [ { "observation_id": "obs-001", "tracklet_id": "t1", "...": "..." } ],
  "supporting_records": []
}
```

**Response** (shortened)

```json
{
  "snapshot_id": "stage_1",
  "model_version": "v1.0.0",
  "config": { "...": "..." },
  "baseline": { "t1": { "status": "accepted", "successor": "t2",
                        "ranked_candidates": [], "reason": "..." } },
  "learned":  { "...": "..." },
  "tracklets": { "t1": [ { "lat": 37.7, "lon": -122.4, "observed_at": "...", "raw": {} } ] }
}
```

**Errors:** always the same shape.

```json
{ "error": { "code": "validation_error",
             "message": "conflicting content for observation_id obs-017",
             "observation_id": "obs-017" } }
```

`api/main.py`

```python
import json, joblib
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.load import ValidationError, read_jsonl
from src.pipeline import run_association

ART = Path("artifacts")
app = FastAPI(title="Track Association")
state = {}


@app.on_event("startup")
def load_artifacts():
    try:
        state["model"] = joblib.load(ART / "model.joblib")
        state["config"] = json.load(open(ART / "config.json"))
    except FileNotFoundError:
        state.clear()          # /health will report it


class AssociateRequest(BaseModel):
    snapshot_id: str
    records: list[dict]
    supporting_records: list[dict] = []


@app.exception_handler(ValidationError)
def handle_validation(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"error": {
        "code": "validation_error", "message": str(exc),
        "observation_id": exc.observation_id}})


@app.get("/health")
def health():
    ok = "model" in state
    return JSONResponse(status_code=200 if ok else 503, content={
        "status": "ok" if ok else "model_not_loaded",
        "model_version": state.get("config", {}).get("model_version")})


@app.post("/tracklets/associate")
def associate(req: AssociateRequest):
    if "model" not in state:
        return JSONResponse(status_code=503, content={"error": {
            "code": "model_not_loaded", "message": "run python -m scripts.train first"}})
    if not req.records:
        return JSONResponse(status_code=422, content={"error": {
            "code": "empty_input", "message": "records must not be empty"}})

    tracklets, out = run_association(req.records, scenario_of,
                                     state["model"], state["config"])
    return {
        "snapshot_id": req.snapshot_id,
        "model_version": state["config"]["model_version"],
        "config": state["config"],
        "baseline": out["baseline"],
        "learned": out["learned"],
        "tracklets": {tid: [{"lat": o["lat"], "lon": o["lon"],
                             "observed_at": o["observed_at"].isoformat(),
                             "alt_ft": o["alt_ft"], "raw": o["raw"]} for o in obs]
                      for tid, obs in tracklets.items()},
    }


# Helper for the UI: list and load supplied scenarios / snapshots
DATASETS = {"stage_1": "data/staged/snapshot_1.jsonl",
            "stage_2": "data/staged/snapshot_2.jsonl",
            "validation": "data/val/observations.jsonl"}   # ASSUMPTION: paths


@app.get("/datasets")
def list_datasets():
    return list(DATASETS)


@app.get("/datasets/{name}")
def get_dataset(name: str):
    if name not in DATASETS:
        return JSONResponse(status_code=404, content={"error": {
            "code": "not_found", "message": f"unknown dataset {name}"}})
    return read_jsonl(DATASETS[name])


app.mount("/", StaticFiles(directory="api/static", html=True), name="ui")
```

> Supporting records: join them with the keys documented in the data package.
> If you decide not to use them, accept them in the request, ignore them, and say why in the README.

```bash
uvicorn api.main:app --reload --port 8000
# open http://localhost:8000
```

The model is loaded from disk at startup, so a restart is safe. No training happens during requests.

---

## 16. Step 13 — Web interface

One HTML file. A plain SVG plot means no external libraries, and everything works offline.

`api/static/index.html`

```html
<!doctype html>
<html>
<head>
<meta charset="utf-8"><title>Track Association</title>
<style>
  body { font-family: sans-serif; margin: 16px; }
  table { border-collapse: collapse; margin-top: 8px; }
  td, th { border: 1px solid #ccc; padding: 4px 8px; font-size: 13px; }
  .error { color: #b00; } .accepted { color: #070; } .unresolved { color: #a60; }
  svg { border: 1px solid #ccc; background: #fafafa; }
</style>
</head>
<body>
<h2>Aircraft Track Association</h2>

<select id="dataset"></select>
<select id="method"><option>baseline</option><option>learned</option></select>
<button onclick="run()">Run association</button>
<button onclick="compareStages()">Compare stage 1 vs 2</button>
<div id="status"></div>

<svg id="plot" width="700" height="450"></svg>
<div id="tables"></div>
<pre id="detail"></pre>

<script>
let last = null;

async function init() {
  const res = await fetch("/datasets");
  const names = await res.json();
  document.getElementById("dataset").innerHTML =
    names.map(n => `<option>${n}</option>`).join("");
}

function setStatus(text, isError = false) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = isError ? "error" : "";
}

async function associate(name) {
  const records = await (await fetch(`/datasets/${name}`)).json();
  const res = await fetch("/tracklets/associate", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({snapshot_id: name, records})
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error?.message || "request failed");
  return body;
}

async function run() {
  setStatus("Loading...");
  try {
    last = await associate(document.getElementById("dataset").value);
    if (Object.keys(last.tracklets).length === 0) { setStatus("No tracklets in input."); return; }
    setStatus(`Done. Model ${last.model_version}`);
    render();
  } catch (e) { setStatus("Error: " + e.message, true); }
}

function render() {
  const method = document.getElementById("method").value;
  const res = last[method];
  drawPlot(last.tracklets, res);
  drawTable(res);
}

function drawPlot(tracklets, res) {
  const pts = Object.values(tracklets).flat();
  const lats = pts.map(p => p.lat), lons = pts.map(p => p.lon);
  const [minLat, maxLat, minLon, maxLon] =
    [Math.min(...lats), Math.max(...lats), Math.min(...lons), Math.max(...lons)];
  const x = lon => 20 + (lon - minLon) / (maxLon - minLon || 1) * 660;
  const y = lat => 430 - (lat - minLat) / (maxLat - minLat || 1) * 410;

  let svg = "";
  for (const [tid, obs] of Object.entries(tracklets)) {
    const color = res[tid]?.status === "accepted" ? "#070" : "#a60";
    svg += `<polyline fill="none" stroke="${color}" stroke-width="2"
              points="${obs.map(o => `${x(o.lon)},${y(o.lat)}`).join(" ")}"
              onclick="showDetail('${tid}')"><title>${tid}</title></polyline>`;
    const e = obs[obs.length - 1];
    svg += `<text x="${x(e.lon) + 3}" y="${y(e.lat)}" font-size="10">${tid}</text>`;
  }
  for (const [tid, r] of Object.entries(res)) {      // dashed line = accepted link
    if (!r.successor) continue;
    const a = tracklets[tid].at(-1), b = tracklets[r.successor][0];
    svg += `<line x1="${x(a.lon)}" y1="${y(a.lat)}" x2="${x(b.lon)}" y2="${y(b.lat)}"
              stroke="#00f" stroke-dasharray="4"/>`;
  }
  document.getElementById("plot").innerHTML = svg;
}

function drawTable(res) {
  let html = `<table><tr><th>Tracklet</th><th>Status</th><th>Successor</th>
              <th>Top candidates (score)</th><th>Reason</th></tr>`;
  for (const [tid, r] of Object.entries(res)) {
    const cands = r.ranked_candidates.map(c => `${c.candidate} (${c.score})`).join(", ");
    html += `<tr onclick="showDetail('${tid}')"><td>${tid}</td>
             <td class="${r.status}">${r.status}</td><td>${r.successor || "-"}</td>
             <td>${cands || "-"}</td><td>${r.reason}</td></tr>`;
  }
  document.getElementById("tables").innerHTML = html + "</table>";
}

function showDetail(tid) {
  const method = document.getElementById("method").value;
  document.getElementById("detail").textContent = JSON.stringify({
    result: last[method][tid], original_reports: last.tracklets[tid].map(o => o.raw)
  }, null, 2);
}

async function compareStages() {
  setStatus("Loading both stages...");
  try {
    const [s1, s2] = await Promise.all([associate("stage_1"), associate("stage_2")]);
    const method = document.getElementById("method").value;
    let html = `<h3>Stage 1 vs Stage 2 (${method})</h3><table>
                <tr><th>Tracklet</th><th>Stage 1</th><th>Stage 2</th><th>Stage 2 reason</th></tr>`;
    const ids = new Set([...Object.keys(s1[method]), ...Object.keys(s2[method])]);
    for (const tid of [...ids].sort()) {
      const a = s1[method][tid]?.successor || "-", b = s2[method][tid]?.successor || "-";
      const style = a !== b ? ' style="background:#ffe"' : "";
      html += `<tr${style}><td>${tid}</td><td>${a}</td><td>${b}</td>
               <td>${s2[method][tid]?.reason || "new tracklet"}</td></tr>`;
    }
    document.getElementById("tables").innerHTML = html + "</table>";
    last = s2; drawPlot(s2.tracklets, s2[method]);
    setStatus("Done. Highlighted rows changed.");
  } catch (e) { setStatus("Error: " + e.message, true); }
}

document.getElementById("method").onchange = () => last && render();
init();
</script>
</body>
</html>
```

This covers: scenario select, lat/lon plot, accepted and unresolved tracklets, candidates,
baseline vs learned, stage comparison, original reports, and loading/empty/success/error states.

---

## 17. Step 14 — Tests

`tests/test_load.py`

```python
import copy, random, pytest
from src.load import load_observations, ValidationError

def rec(obs_id="o1", tid="t1", t="2026-01-10T10:00:00Z", lat=10.0, lon=20.0, alt=None):
    return {"observation_id": obs_id, "tracklet_id": tid, "observed_at": t,
            "received_at": "2026-01-10T10:05:00Z", "latitude": lat, "longitude": lon,
            "altitude_ft": alt, "source": "feed-a", "attributes": {"x": [1, 2]}}

def test_missing_required_field():
    r = rec(); del r["source"]
    with pytest.raises(ValidationError):
        load_observations([r])

def test_empty_id():
    with pytest.raises(ValidationError):
        load_observations([rec(obs_id="")])

def test_bad_latitude():
    with pytest.raises(ValidationError):
        load_observations([rec(lat=95)])

def test_non_utc_timestamp():
    with pytest.raises(ValidationError):
        load_observations([rec(t="2026-01-10T10:00:00+05:30")])

def test_identical_duplicate_kept_once():
    r = rec()
    reordered = dict(reversed(list(copy.deepcopy(r).items())))   # different key order
    out = load_observations([r, reordered])
    assert len(out["t1"]) == 1

def test_conflicting_duplicate_rejected():
    with pytest.raises(ValidationError) as e:
        load_observations([rec(lat=10.0), rec(lat=11.0)])
    assert e.value.observation_id == "o1"

def test_missing_vs_null_are_different():
    a = rec(); b = rec(); del b["altitude_ft"]          # null vs missing
    with pytest.raises(ValidationError):
        load_observations([a, b])

def test_sorted_by_observed_at():
    recs = [rec("o2", t="2026-01-10T10:02:00Z"), rec("o1", t="2026-01-10T10:00:00Z")]
    out = load_observations(recs)
    assert [o["observation_id"] for o in out["t1"]] == ["o1", "o2"]

def test_missing_altitude_stays_none():
    out = load_observations([rec(alt=None)])
    assert out["t1"][0]["alt_ft"] is None
```

`tests/test_decide.py`

```python
from src.decide import decide

F = {"gap_min": 5, "pred_error_km": 1, "dist_km": 1}

def test_one_successor_one_predecessor():
    cands = [("A", "B", F), ("A", "C", F), ("D", "B", F)]
    res = decide(["A", "B", "C", "D"], cands, [0.9, 0.2, 0.8], threshold=0.5, margin=0.0)
    successors = [r["successor"] for r in res.values() if r["successor"]]
    assert len(successors) == len(set(successors))       # no B used twice
    assert res["A"]["successor"] == "B"
    assert res["D"]["successor"] is None

def test_ambiguous_left_unresolved():
    cands = [("A", "B", F), ("A", "C", F)]
    res = decide(["A", "B", "C"], cands, [0.81, 0.80], threshold=0.5, margin=0.1)
    assert res["A"]["status"] == "unresolved"

def test_below_threshold_unresolved():
    res = decide(["A", "B"], [("A", "B", F)], [0.1], threshold=0.5, margin=0)
    assert res["A"]["status"] == "unresolved"
```

`tests/test_pipeline.py`: reordering, duplicates, and save/load.

```python
import random, joblib, json, copy
from src.pipeline import run_association

def test_reorder_and_duplicates_same_result(sample_records, model, config):
    _, r1 = run_association(sample_records, scenario_of, model, config)
    shuffled = copy.deepcopy(sample_records) + copy.deepcopy(sample_records[:5])
    random.Random(1).shuffle(shuffled)
    _, r2 = run_association(shuffled, scenario_of, model, config)
    assert r1["baseline"] == r2["baseline"]
    assert r1["learned"] == r2["learned"]

def test_model_save_load_same_scores(tmp_path, sample_records, model, config):
    joblib.dump(model, tmp_path / "m.joblib")
    loaded = joblib.load(tmp_path / "m.joblib")
    _, r1 = run_association(sample_records, scenario_of, model, config)
    _, r2 = run_association(sample_records, scenario_of, loaded, config)
    assert r1["learned"] == r2["learned"]
```

Put `sample_records`, `model`, and `config` fixtures in `tests/conftest.py`
(a few hand-made tracklets and a tiny model trained on them).

```bash
pytest -q
```

---

## 18. Step 15 — README and submission

### README.md template

```markdown
# Aircraft Track Association

## Commands
pip install -r requirements.txt
python -m scripts.train          # fits model, picks thresholds on validation
python -m scripts.evaluate       # test metrics + stress test
python -m scripts.staged_demo    # stage 1 / stage 2 results + diff
pytest -q
uvicorn api.main:app --port 8000 # UI at http://localhost:8000
Seed: 42

## Approach
- Shared preprocessing and candidate generation
- Baseline: score = -(prediction error km), higher = better
- Learned: gradient boosting on pair features, score 0..1, higher = better, NOT calibrated
- Greedy decision: threshold + ambiguity margin + one-to-one constraints

## Decision policy
Wrong links cost more than unresolved ones, so thresholds were chosen on validation as:
highest recall with precision >= 0.95.

## Field interpretation
(table from Step 2: confirmed vs inferred vs unknown, used / excluded and why)

## Results
(test table: precision, recall, wrong links, unresolved %, candidate recall, both methods)
(validation sweep: strict / chosen / loose)
(stress test result)

## Failure analysis
(3+ cases, including one disagreement)

## Staged evidence
(what was accepted in stage 1, what changed in stage 2, and why)

## Synthetic data caveats
- Straight-line motion model; real aircraft turn, climb, hold
- Simulated noise is probably cleaner than real sensor noise
- Real data needs: real gap distributions, dense traffic near airports,
  identifier reliability, and calibration of scores

## Limitations / unfinished
## Time spent
## AI assistance (see PROMPTS.md)
```

### PROMPTS.md

Copy **every** prompt you sent to any AI tool, in order, including this conversation.
Add one line per prompt on how you used the answer.

---

## 19. Time plan

| Hours | Work |
|---|---|
| 0:00–1:00 | Read data package, run `explore_fields.py`, fill the field table |
| 1:00–2:30 | `load.py`, `geo.py`, `features.py` + load tests |
| 2:30–3:30 | Candidates + baseline + decide + metrics, end to end on validation |
| 3:30–4:30 | Learned model, `train.py`, threshold sweep |
| 4:30–5:15 | `evaluate.py`, stress test, failure analysis |
| 5:15–5:45 | Staged demo |
| 5:45–7:00 | API + UI |
| 7:00–8:00 | Remaining tests, README, PROMPTS.md |

If time runs out: cut UI polish first, never the evaluation or the README honesty.

---

## 20. Final checklist

- [ ] Validation: required fields, empty IDs, UTC, lat/lon ranges
- [ ] Identical duplicates removed, conflicting IDs rejected with ID in error
- [ ] Sorted by `observed_at`; `received_at` used only for snapshots
- [ ] Null altitude kept as missing, never 0
- [ ] Raw records preserved and visible in the UI
- [ ] No IDs, labels or scenario IDs used as features
- [ ] Candidate recall reported
- [ ] Baseline and learned both produce accepted / unresolved + ranked candidates + reasons
- [ ] Thresholds chosen on validation only; test run once
- [ ] Precision, recall, wrong links, unresolved %, candidate recall for both methods
- [ ] Conservative / chosen / permissive comparison
- [ ] Stress test without retraining
- [ ] 3+ failure cases, including a disagreement
- [ ] Stage 1 and stage 2 results saved with snapshot id, model version, config
- [ ] `POST /tracklets/associate` and `GET /health` with consistent JSON errors
- [ ] Model loads from disk after restart; no training in requests
- [ ] UI: plot, tables, compare methods, compare stages, all states
- [ ] Tests: validation, duplicates, conflicts, ordering, null altitude, constraints, save/load, reorder
- [ ] README: assumptions, fields used/excluded, limitations, time spent
- [ ] PROMPTS.md with every AI prompt
# Execution Plan — Aircraft Track Association

Planner: Claude Opus 5.5 (main session). Coding: Opus 5.5 for the data, ML and backend core; a Claude Sonnet 5.5 subagent for the self-contained web UI, built against the API contract in §9.

## 0. Situation and key assumption

`original_question.md` describes the **Track Association** challenge. The supplied `Required_files/` packet belongs to a different challenge (Movement History: identity mappings, replay of HTTP requests) and contains **no tracklets, partitions, labels or staged snapshots**.

The user told us to proceed on stated assumptions. So:

| # | Assumption | Consequence |
|---|---|---|
| A1 | The missing data package is replaced by a **seeded synthetic generator** (`scripts/generate_data.py`, seed 42) that produces every artefact the brief lists. | All results are synthetic-on-synthetic. The README says so loudly. |
| A2 | The generator's ground truth (aircraft IDs) is written **only** to label files and a `_truth/` folder that no model code reads. | Labels stay separate from inputs. |
| A3 | Rules from `Required_files/Clarifications.docx` that apply generally are reused: UTC RFC3339 with `Z` or `+00:00` and ≤3 fractional digits; leap seconds and invalid dates rejected; JSON equality ignores key order, keeps array order, treats different JSON types (e.g. `true` vs `1`) as different, allows `1 == 1.0`; missing ≠ null; finite numbers only for coordinates. | Implemented in `src/load.py`; covered by tests. |
| A4 | Scenario membership is supplied in `tracklet_index.jsonl` (`tracklet_id → scenario_id`) and is used **only** to scope processing. | API requests may carry an optional `tracklet_scenarios` map; if it is absent, the request is treated as one scenario. |
| A5 | The generator is written *before* the models and is never tuned to make a model look good. Its difficulty knobs are fixed in one place and recorded in `data/README_source_notes.md`. | Honest evaluation within the synthetic world. |

Because I write the generator, I know the "true" meaning of every optional field. To keep the field investigation honest, the **source notes deliberately document only part of each field**. The README's field table records what was *inferred from distributions* (the exploration script output), and keeps that separate from what the generator code says.

## 1. Synthetic data package (`data/`)

```
data/
  README_source_notes.md        # sources, units, join keys, movement assumptions, partial field docs
  schema.json                   # core observation schema
  train/ val/ test/
    observations.jsonl          # position reports (shuffled, with duplicates, delayed receipts)
    tracklet_context.jsonl      # supporting stream, joined by tracklet_id
    id_reports.jsonl            # supporting stream, joined by observation_id
    tracklet_index.jsonl        # tracklet_id -> scenario_id (grouping only)
    labels.jsonl                # {tracklet_id, next_tracklet_id|null}  (kept separate)
  staged/
    snapshot_1/ snapshot_2/     # same 5 files each, filtered by received_at <= cutoff
    snapshots.json              # cutoffs
  _truth/                       # aircraft ids per tracklet (debug only, never read by src/)
```

**World model:** each scenario is a 3 h window holding 6–12 aircraft. Each aircraft flies at 300–520 kt with piecewise-constant heading and gentle turns (≤ 2°/s), plus some climbs and descents. Each aircraft's track is cut into 1–5 tracklets separated by gaps of 1–25 min.

**Difficulty built in** (matching the brief's list):
- clear continuations;
- *parallel pairs*: two aircraft 4–12 km apart with the same heading and speed;
- *crossing traffic*;
- turns inside gaps;
- dropped tracklets, which leave **no successor**;
- aircraft whose last tracklet has no successor;
- irregular sampling (4–40 s);
- position noise that depends on the source;
- rare position glitches;
- identical duplicates (~1.5 %, some with keys reordered);
- shuffled file order;
- receipt delays.

**Sources and attributes** (one source per tracklet):
- `feed-a`: `gs` (knots), `trk` (deg), `sq` (4-digit octal code, a reported identifier that is sometimes reassigned), `rssi` (irrelevant).
- `feed-b`: `spd` (knots, units undocumented), `c2` (track deg + noise), `ref` (6-hex reported identifier, ~8 % missing, ~4 % wrong), `q` (quality class correlated with noise), `node` (ingest node, irrelevant).
- `feed-c`: `spd` (**m/s**, same name, different unit), `hdg`, `vr` (ft/min), `alt_ref` (`baro`/`geo`). Altitude is null more often in this feed.

**Supporting streams:**
- `tracklet_context`: `cat` (aircraft category; mostly consistent per aircraft, sometimes missing, sometimes conflicting), `site` (irrelevant).
- `id_reports`: `cs` (callsign, joined per observation; partial coverage; ~5 % of aircraft change callsign; occasionally a callsign is shared incorrectly).

**Labels:** the next tracklet of the same aircraft *present in that input*, else null.

**Staged scenario:** a separate scenario in which some whole tracklets (and some tails of tracklets) have receipt delays of 30–90 min. Snapshot 1 = records with `received_at ≤ T1`; snapshot 2 = records with `received_at ≤ T2`. Each snapshot gets its own labels, so the A→C link in snapshot 1 can become A→B→C in snapshot 2.

## 2. Pipeline modules (`src/`)

| Module | Responsibility |
|---|---|
| `load.py` | Validate (A3 rules), canonical-JSON dedupe, reject conflicts naming the ID, group by tracklet, sort by `(observed_at, observation_id)`, keep raw. Join supporting streams by their documented keys, with the same dedupe and conflict rules. |
| `geo.py` | haversine, bearing, angle diff, forward/backward projection. |
| `features.py` | Tracklet summary: robust start/end velocity from a least-squares fit over a ~3 min window; glitch flagging (consecutive implied speed > 1500 km/h, excluded from the fit, raw value preserved); start/end altitude with missing kept as None; reported identifiers per tracklet; source-normalised attribute speeds. Pair features: gap, distance, implied speed, forward and backward prediction error, speed diff, heading change, bearing alignment, altitude diff (NaN if unknown), callsign match / category match (NaN if unknown). |
| `candidates.py` | B starts strictly after A ends; same scenario; gap ≤ G; implied speed ≤ V; forward or backward prediction error ≤ E·(1+gap). G, V and E are chosen from **train** true-link distributions to keep ~99.5 %, then confirmed on **val** and frozen. |
| `scorers.py` | Baseline: `score = −(fwd_err + back_err)/2` km (higher = better; not a probability). Learned: `HistGradientBoostingClassifier` on pair features (NaN-native); score = model output in 0..1, higher = better, **not calibrated** (a reliability table on val is reported, but no calibration claim is made). |
| `decide.py` | Deterministic greedy, best pair first: threshold → one-to-one conflict check → ambiguity margin on both A's successors and B's predecessors → accept. Records per-tracklet status, ranked top-5 candidates with measurements, and a reason. |
| `evaluate.py` | Metrics exactly as the brief defines them; zero denominators → `null` plus counts; evaluated over all tracklets, including links lost in candidate generation. |
| `pipeline.py` | `run_association(observations, supporting, scenario_map, model, config)` → results + manifest (input hash over canonical deduped records, model version, config hash). |

## 3. Training and selection (`scripts/train.py`)

1. Fit on **train**: candidate settings, learned model.
2. On **val**: select the feature set (physics-only vs physics + supporting) and the threshold/margin grid for both methods.
   Rule, fixed now: *maximise recall subject to val precision ≥ 0.98; ties go to the more conservative setting*. Wrong links cost more than unresolved ones.
3. Save `artifacts/model.joblib`, `artifacts/config.json` (version, seed, feature list, candidate settings, decision settings, selection rule) and `results/validation_sweep.json`.

## 4. Evaluation (`scripts/evaluate.py`)

- Run on **test** once, at the frozen settings, for both methods.
- Report conservative / chosen / permissive settings, using the **val** sweep.
- Stress tests on test, with no retraining:
  - **S1** thin observations to every 3rd report, keeping each tracklet's first and last;
  - **S2** remove the `id_reports` supporting stream.
- Disagreements and error cases are dumped to `results/test_errors.json`.
- `results/failure_analysis.md` is written by hand from that dump: 3+ cases, including a disagreement.

## 5. Staged demo (`scripts/staged_demo.py`)

Analyse snapshot 1 and snapshot 2 independently with the same artefacts. Save `results/stage_1.json`, `results/stage_2.json` and `results/staged_diff.json`. The diff lists new, withdrawn and replaced links with measurements and an explanation that distinguishes an "intermediate fragment arrived" from a "different aircraft". Each stage is evaluated against its own labels.

## 6. Backend (`api/main.py`, FastAPI)

- `GET /health`: 200 with model version when the artefacts are loaded, else 503.
- `POST /tracklets/associate`: see §9. Errors use one shape: `{"error": {"code", "message", "details"}}`. Status 422 for invalid records or conflicts, 400 for malformed bodies, 503 when there is no model.
- Helper endpoints for the UI: `GET /datasets`, `GET /datasets/{name}` (returns a ready-made request payload for a supplied scenario or snapshot), `GET /results/staged`.
- No training happens at request time. Each result is saved with a manifest to `results/api_runs/`.

## 7. Web UI (`api/static/index.html`, Sonnet subagent)

A single file of plain JS and SVG with no external dependencies. It covers: dataset select → run; lat/lon plot with accepted links drawn; tracklet table (status, successor, candidates, reason); baseline vs learned side by side with disagreements highlighted; stage 1 vs stage 2 diff view; a detail pane showing measurements and original reports; and loading, empty, success and error states.

## 8. Tests (`tests/`)

Validation (required fields, empty IDs, UTC formats, coordinate ranges, non-finite values, bool-as-number), duplicates (key order, 1 vs 1.0, true vs 1, missing vs null, array order), conflicting IDs, ordering, missing altitude, decision constraints (one successor, one predecessor, strict time order, no cycles, ambiguity → unresolved), reorder and duplicate invariance of end-to-end results, model save/load consistency, and API (health, success, error shapes).

## 9. API contract (shared with the UI subagent)

Request:
```json
{ "snapshot_id": "test/scn-t-003",
  "observations": [ {core observation} ],
  "supporting": { "tracklet_context": [ ... ], "id_reports": [ ... ] },
  "tracklet_scenarios": { "trk-...": "scn-..." } }
```
Response:
```json
{ "manifest": { "snapshot_id", "input_hash", "model_version", "config_hash", "decision_config", "generated_at", "n_observations_in", "n_observations_unique", "n_tracklets" },
  "tracklets": { "<tid>": { "scenario_id", "source", "n_points", "start_time", "end_time",
                            "points": [ {"lat","lon","alt_ft","observed_at","observation_id","glitch"} ],
                            "raw": [ original records ], "summary": { ... } } },
  "methods": { "baseline": { "<tid>": RESULT }, "learned": { "<tid>": RESULT } },
  "metrics": null }
```
`RESULT = { "status": "accepted"|"unresolved", "successor", "predecessor", "score", "reason", "ranked_candidates": [ {"candidate","score","rank","measurements":{...}} ] }`

## 10. Order of work

1. Generator → data. 2. `load`, `geo`, `features` + exploration script. 3. Candidates, baseline, decide, metrics; validate end to end on val. 4. Learned model + `train.py`. 5. `evaluate.py` and stress tests. 6. Staged demo. 7. API (UI subagent runs in parallel). 8. Tests. 9. README, failure analysis, field table, PROMPTS.md.

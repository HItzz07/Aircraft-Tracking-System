# Aircraft Track Association

This project associates disconnected aircraft tracklets (A → B means "B is the next observed fragment of the same aircraft"). It compares a physics baseline with a learned scorer, leaves uncertain cases unresolved, and shows how delayed evidence changes earlier decisions. It ships with a FastAPI backend and a small web UI.

> **Read this first — synthetic data assumption.** The challenge's data package (partitions, supporting streams, labels, staged snapshots) was **not** among the supplied files. `Required_files/` holds a different challenge's packet (identity-mapping ingestion), which has no tracklets or labels.
>
> As instructed, I proceeded on stated assumptions and wrote a seeded generator (`scripts/generate_data.py`) that produces an equivalent package. **Every number below is synthetic-on-synthetic.** See [PLAN.md](PLAN.md) §0 for the full assumption list (A1–A5).
>
> Two rules from `Required_files/Clarifications.docx` were reused where they apply: the UTC timestamp format and JSON identity for duplicates.

## Commands (Windows PowerShell; on Linux/macOS use `.venv/bin/python`)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

.venv\Scripts\python.exe -m scripts.generate_data      # 1. data/ (seed 42, ~11 s)
.venv\Scripts\python.exe -m scripts.explore_fields     # 2. results/field_exploration.json (train only)
.venv\Scripts\python.exe -m scripts.train              # 3. artifacts/ + results/validation_sweep.json (~1 min)
.venv\Scripts\python.exe -m scripts.evaluate           # 4. results/test_metrics.json, test_errors.json
.venv\Scripts\python.exe -m scripts.staged_demo        # 5. results/stage_1.json, stage_2.json, staged_diff.json
.venv\Scripts\python.exe -m pytest -q                  # 6. 77 tests, ~10 s
.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000   # 7. UI at http://localhost:8000
```

`data/` (except the source notes) is git-ignored because it is 88 MB and step 1 regenerates it exactly. `artifacts/` and `results/` are committed, so the service and saved results work without retraining (the data is still needed for `/datasets` and evaluation). Everything is seeded (`SEED = 42`) and runs locally on CPU. Tested with Python 3.13, scikit-learn 1.9.1 and FastAPI 0.162. `scripts/diagnose.py` is a development aid that categorises wrong links on **val**.

## Layout

| Path | What |
|---|---|
| `src/load.py` | validation, canonical-JSON dedupe, conflict rejection, ordering, supporting joins |
| `src/features.py` | glitch flagging, robust end/start velocity fits, tracklet summaries, pair features |
| `src/candidates.py` | candidate generation, shared by both methods |
| `src/scorers.py` | baseline score, learned model (HistGradientBoosting) |
| `src/decide.py` | greedy decisions: threshold, skip check, conflict, ambiguity margin; constraint checker |
| `src/evaluate.py` | metrics exactly as defined in the brief |
| `src/pipeline.py` | one entry point used by train, evaluate, demo and API; manifest and input hash |
| `api/main.py`, `api/static/index.html` | FastAPI service and single-file UI |
| `results/` | all saved evaluation outputs, [failure analysis](results/failure_analysis.md), API run logs |
| `data/README_source_notes.md` | the (partial) source documentation of the data package |

## Data preparation

- **Validation:** a record is rejected (HTTP 422, naming the observation ID) when:
  - a required field is missing, null or empty;
  - an ID or `source` is not a string (IDs are case-sensitive and never trimmed);
  - a timestamp is not RFC3339 UTC (`Z` or `+00:00`, ≤ 3 fractional digits). Other offsets, naive times, impossible dates and leap seconds are rejected;
  - lat/lon are outside [−90, 90] / [−180, 180], or are not finite JSON numbers (strings, booleans, NaN and Inf are rejected);
  - `altitude_ft` is not a number or null, or `attributes` is not an object.
- **Duplicates:** records with the same `observation_id` are compared in **canonical JSON** form:
  - key order is ignored and array order kept;
  - `true` ≠ `1`, but `1` = `1.0`;
  - a missing field ≠ an explicit null;
  - comparison happens on the original record, before any transformation.

  Identical copies are collapsed. A different copy rejects the **whole request** with `conflicting_id` and the ID. The same rules apply to `context_id` and `report_id` in the supporting streams.
- **Times:** `observed_at` drives ordering and all movement features. `received_at` is used only for snapshot availability (and reported as `max_receipt_delay_s`). Reports are sorted by `(observed_at, observation_id)`. The ID only breaks exact ties, so ordering is deterministic.
- **Irregular sampling:** every feature uses real elapsed time, never point counts. End and start velocities come from a least-squares constant-velocity fit over the last/first 3 minutes (at least 3 points). This also smooths position noise.
- **Surprising but valid values:** a point is flagged as a *glitch* when both reaching and leaving it would need more than 1,500 km/h. Glitches are excluded from fits but kept, and shown as ✕ in the UI. 387 were flagged in train.
- **Missing values:** a missing altitude stays `None`/NaN and is never treated as 0. HistGradientBoosting handles NaN natively. The baseline does not use altitude.
- **Preservation:** each original record is kept verbatim (`raw`) and returned by the API and UI.
- **Labels** live in separate files. They are read only by `scripts/train.py` (train labels for fitting, val labels for selection), `scripts/evaluate.py` and the staged demo. The API attaches labels only *after* decisions, and only for display.

## Field interpretation

All evidence comes from `results/field_exploration.json`, computed on **train**. **Documented** means stated in the source notes. **Inferred** means established from distributions, not from documentation.

| Source / field | Documented | Evidence from data | Interpretation | Status | Used? |
|---|---|---|---|---|---|
| core time, lat/lon, `altitude_ft` | yes | — | as documented | confirmed | **yes** |
| feed-a `gs` | "ground speed" | computed km/h ÷ gs = **1.850** (p25–p75 1.844–1.856) | knots | inferred | no (redundant with fitted velocity; kept in summaries as `reported_speed_kmh`) |
| feed-b `spd` | "speed", no unit | ratio **1.851** | knots | inferred | no (same reason) |
| feed-c `spd` | "speed" | ratio **3.598** | **m/s**: same name as feed-b, different unit | inferred | no (same reason) |
| feed-a `trk`, feed-b `c2`, feed-c `hdg` | partly | field minus computed heading: median ≈ 0°, IQR ±1.1° / ±3.1° / ±2.4° | track angle in degrees | inferred | no (redundant) |
| feed-b `q` | "quality class" | fit residual median 86 / 157 / 366 m for q = 1/2/3 | position-quality grade, 3 = worst | inferred | no (noise already shows in the fit; listed as next step) |
| feed-c `alt_ref` | "reference tag" | `geo` tracklets read **+530 ft** above every other source on true links (IQR 445–589) | different vertical datum | inferred | **yes**: `geo` altitudes are excluded from altitude comparisons; no correction is applied |
| feed-c `vr` | no | symmetric around 0, ±1,000 ft/min, 20 % absent | vertical rate, ft/min | inferred | no |
| feed-b `ref` | "upstream identifier" | same on 294 / different on 21 / missing on 2,281 true links | reported ID: mostly stable, sometimes wrong, feed-b only | inferred | **yes**, as a 1/0/NaN `ref_match` feature (learned only) |
| feed-a `sq` | "broadcast code" | 3 values (1200, 2000, 7000) cover 6.7 k of 78 k reports; 17/398 true links change code | shared or reassigned code: unreliable | inferred | **no**: collisions make equality weak, and it can be reassigned |
| feed-a `rssi`, feed-b `node`, context `site`, `fw`, id `src` | internal | no relationship to motion | infrastructure | assumption | **no** (irrelevant) |
| `tracklet_context.cat` | "category" | same on 1,683 / different on 160 true links; 71 tracklets have conflicting records | type category, noisy, only 5 values | inferred | **yes**, as `cat_match` (NaN if missing or conflicting) |
| `id_reports.cs` | "callsign, not authoritative" | same 1,005 / different 60 / missing 1,531 true links | reported callsign | documented + inferred | **yes**, as `callsign_match` (majority per tracklet; NaN if > 20 % disagree) |

**Availability at inference:** the reported identifiers (`cs`, `ref`, `cat`) arrive in the same snapshot as the reports, through documented join keys. They are never used as labels, and the learned model treats them as soft evidence, not identity.

**Unresolved assumptions:**
- the meaning of `q` levels beyond "higher is noisier";
- whether `geo` is a fixed offset everywhere (we exclude it rather than correct it);
- whether `sq` changes signal anything.

**Do the supporting streams help?** Yes, modestly, and only for the learned model:
- On **val** (selection), adding `callsign_match`/`cat_match`/`ref_match` cut wrong links from 8 to 1 and raised recall from 0.957 to 0.974.
- On **test**, removing `id_reports` raised learned wrong links from 11 to 18 and lowered recall from 0.953 to 0.935 (see S2 below).
- 7 of the 16 baseline wrong links on test carry an explicit callsign mismatch.
- They can also mislead. In the staged demo, a changed callsign made the learned model withhold a correct link (see [failure analysis](results/failure_analysis.md)).

## Approach

**Candidate generation** is shared by both methods, so the comparison is fair. A pair A → B is kept when all of these hold:
- B starts strictly after A ends;
- both are in the same scenario;
- gap ≤ 60 min;
- implied speed ≤ 1,110 km/h;
- min(forward, backward) projection error ≤ 11.66 km × (2 + gap_min).

The limits are derived from **train** true links (p99.9 / p99.5 with a 10 % allowance) and confirmed on val. The error gate is loose because aircraft can turn by up to 120° inside a gap. Candidate recall is **0.996 train, 0.996 val, 0.990 test**.

**Pair features** (from fitted end/start states; IDs are never features):

| Group | Features |
|---|---|
| Physics | gap, distance, implied speed, forward projection error (A's end velocity → B's start time), backward projection error, speed difference, heading change, bearing alignment, implied-to-fitted speed ratio, altitude difference (NaN if unknown) |
| Supporting | `callsign_match`, `cat_match`, `ref_match` (1 / 0 / NaN) |

A test checks that renaming every tracklet ID leaves all decisions unchanged.

**Baseline (physical continuity):** `score = −mean(fwd_err_km, back_err_km) / (2 + gap_min)`.
- Units are km of mismatch per minute of gap. **Higher is better**; 0 is a perfect straight-line continuation.
- It is not a probability.
- Validation chose the per-minute variant over raw km.

**Learned:** `HistGradientBoostingClassifier` (300 iterations, learning rate 0.05, 15 leaves, L2 1.0, seed 42), trained on train candidate pairs with label 1 for the true immediate successor.
- The score is the model output in [0, 1]. **Higher is better.**
- It is **not calibrated confidence**. A descriptive reliability table on val is in `validation_sweep.json` (`learned_reliability_val`); for example, the 0.3–0.4 bin has a 45 % true rate. I make no calibration claim.

**Decision policy** (`src/decide.py`, the same code for both methods): visit pairs from best to worst score, then for each pair A → B:
1. If A is already linked, skip.
2. If the score is below the threshold, withhold.
3. **Skip check:** if some B′ fits inside the gap (A → B′ and B′ → B both ≥ threshold), withhold A → B as probably not *immediate*. Such skip pairs do not count as rivals for A → B′ or B′ → B.
4. If B already has a stronger predecessor, withhold (conflict).
5. If another *still-available* successor of A, or predecessor of B, is within `margin`, withhold (ambiguous).
6. Otherwise accept.

One successor and one predecessor per tracklet are enforced by steps 1 and 4. Strict time order comes from candidate generation, and there are no cycles because every link moves forward in time. `check_constraints` asserts all three on every run.

Every tracklet gets a status, its successor and predecessor, the top-5 ranked candidates with measurements, and a one-line reason.

**Known limitation:** greedy is not globally optimal. A strong wrong link can block a correct one (failure analysis, cases 1 and 4).

**Cost of errors → selection rule** (fixed before test): on validation, maximise `correct − 5 × incorrect` subject to precision ≥ 0.98. Ties go to the higher threshold and then the larger margin, i.e. the more conservative setting.

The sweep covers threshold × margin × skip check, the baseline variant, and the learned feature set (physics vs physics + supporting).

Chosen settings:

| | Variant / features | Threshold | Margin | Skip check |
|---|---|---|---|---|
| Baseline | per-minute | −3.0 | 0.1 | on |
| Learned | physics + supporting | 0.02 | 0.0 | on |

All of this is frozen in `artifacts/config.json`.

## Results

### Test, at the frozen operating point (run once)

| Method | Precision | Recall | Wrong links | Unresolved (of labelled) | Candidate recall | Accepted / labelled links |
|---|---|---|---|---|---|---|
| Baseline | **0.973** | **0.675** | 16 | 0.311 | 0.990 | 601 / 867 |
| Learned | **0.987** | **0.953** | 11 | 0.038 | 0.990 | 837 / 867 |

Test precision is below the 0.98 floor met on val (0.988 / 0.999), a real generalisation gap that I report rather than retune. The test partition has 1,057 tracklets, 867 of them with a labelled successor. Accepting a successor for a tracklet labelled "none" counts as wrong, and the evaluation covers every tracklet, including links lost in candidate generation.

### Conservative vs permissive (validation, used for selection)

| Method | Setting | Threshold | Precision | Recall | Wrong |
|---|---|---|---|---|---|
| Baseline | permissive | −12.0 | 0.952 | 0.902 | 43 |
| Baseline | **chosen** | −3.0 | 0.988 | 0.701 | 8 |
| Baseline | conservative | −1.5 | 0.995 | 0.628 | 3 |
| Learned | **chosen** (lowest grid value) | 0.02 | 0.999 | 0.974 | 1 |
| Learned | conservative | 0.1 | 0.999 | 0.937 | 1 |
| Learned | stricter | 0.5 | 1.000 | 0.738 | 0 |

The full grid is in `results/validation_sweep.json`. For the learned model, the chosen threshold sits at the bottom of the grid: validation never penalised permissiveness. On test, though, the low-score acceptances (< 0.06) produced 5 of its 11 wrong links. In hindsight a margin of 0.05 would have been safer; I did **not** change it after seeing test.

### Stress tests on test (labels, model and settings unchanged; no retraining)

| Test | Baseline P / R / wrong | Learned P / R / wrong |
|---|---|---|
| none | 0.973 / 0.675 / 16 | 0.987 / 0.953 / 11 |
| **S1** keep every 3rd report (first and last kept) | 0.972 / 0.676 / 17 | 0.983 / 0.950 / 14 |
| **S2** remove the `id_reports` stream | 0.973 / 0.675 / 16 | 0.978 / 0.935 / 18 |
| S2b remove all supporting streams | 0.973 / 0.675 / 16 | 0.979 / 0.932 / 17 |
| S3 add 300 m Gaussian position noise | 0.973 / 0.669 / 16 | 0.987 / 0.949 / 11 |

Both methods are robust to thinning and extra noise, because the 3-minute velocity fit absorbs them. The learned model degrades gracefully without identifiers, as missing values become NaN, never 0.

### Failure analysis

See [results/failure_analysis.md](results/failure_analysis.md). It covers five cases: a parallel neighbour accepted by both methods; a learned-only wrong link through a 118° turn (a disagreement); a weak link for a tracklet that has no successor; a skip-check false alarm; and long-gap turns that both methods miss.

## Updating associations (staged demo)

Run with `python -m scripts.staged_demo`. Snapshot 1 (received ≤ 11:30Z) and snapshot 2 (≤ 13:30Z) are analysed **independently**, with the same artifacts. Each is scored against its own labels.

| | Stage 1: accepted / unresolved, P / R | Stage 2: P / R |
|---|---|---|
| Baseline | 28 / 33, 1.00 / 0.56 | 1.00 / 0.58 |
| Learned | 48 / 13, 1.00 / 0.96 | 1.00 / 0.94 |

What changed, with measurements from `results/staged_diff.json`:

- **Learned — two links replaced** because delayed intermediate fragments arrived:
  - `trk-8eb8d946dc → trk-6a7bebd8c8` (stage 1: gap 17.5 min, mean error 174 km after an 86° turn, score 0.32) became `trk-8eb8d946dc → trk-0b57de4011 → trk-6a7bebd8c8` (gaps 4.3 and 6.4 min, scores 0.965 and 0.992).
  - `trk-4e94fbf3f4 → trk-2acf0966a9` (stage 1: score 0.401, with a runner-up at 0.401, an effective tie) became `trk-4e94fbf3f4 → trk-4daa5b59ad → trk-2acf0966a9` (scores 0.552 and 0.965).
  - The skip check withdrew the old direct links. **This is the same inferred aircraft with a newly available middle piece, not a change of identity.** The stage-2 labels confirm both chains.
- **Baseline:** it had left those two cases unresolved in stage 1 (the turn put them below threshold). In stage 2 it accepted both new short links (B → C), but still not A → B.
- **Baseline — one new link:** it added `trk-8a21a02f73 → trk-018b767f69`, a final fragment that was entirely delayed (gap 25 min, mean error 4.3 km, heading change 0.1°). The **learned model left it unresolved**: `callsign_match = 0` and a 4,450 ft altitude change outweighed near-perfect kinematics. This is a justified concern about over-trusting identifiers (failure analysis).
- **`trk-aab05f6f4f`:** its delayed tail extended it from 34 to 58 points, shrinking the gap from 11.6 to 6.8 min. The decision was unchanged, and its score rose from 0.62 to 0.99.

Did the evidence justify the changes? Yes, for the replacements: the intermediate fragments fit both sides with small gaps. The learned model's withheld final link is a limitation, not a forced change. Both result sets record `snapshot_id`, `input_hash`, `model_version`, `config_hash`, the full decision config and the receipt cutoff.

## Backend API

| Endpoint | Behaviour |
|---|---|
| `GET /health` | 200 `{"status":"ok","model_version":"v1.0.0",…}`, or 503 `model_not_loaded` |
| `POST /tracklets/associate` | runs baseline and learned on the supplied snapshot and returns the results |
| `GET /datasets`, `GET /datasets/{name}` | UI helpers that list and return ready-made request payloads (val/test scenarios, staged snapshots) |
| `GET /results/staged` | the saved staged diff |

**Request**:
```json
{ "snapshot_id": "test/scn-test-003",
  "observations": [ { "observation_id": "...", "tracklet_id": "...", "observed_at": "...Z", "received_at": "...Z",
                      "latitude": 0, "longitude": 0, "altitude_ft": null, "source": "feed-b", "attributes": {} } ],
  "supporting": { "tracklet_context": [ ... ], "id_reports": [ ... ] },
  "tracklet_scenarios": { "<tracklet_id>": "<scenario_id>" },
  "evaluate_against": "test/scn-test-003" }
```
`supporting`, `tracklet_scenarios` (default: one scenario) and `evaluate_against` (display-only metrics, applied after decisions) are optional.

**Response**:
```json
{ "manifest": { "snapshot_id", "input_hash", "model_version", "config_hash", "decision_config", "generated_at", "n_observations_in", "n_observations_unique", "n_duplicates_removed", "n_tracklets", "n_candidate_pairs" },
  "methods": { "baseline": { "<tid>": RESULT }, "learned": { "<tid>": RESULT } },
  "tracklets": { "<tid>": { "points", "raw", "summary", "supporting", "source", "start_time", "end_time" } },
  "metrics": null, "labels": null }
```
`RESULT = {status: accepted|unresolved, successor, predecessor, score, n_candidates, ranked_candidates[{candidate, score, rank, measurements}], reason}`.

**Errors**: `{"error": {"code", "message", "details"}}`.
- 400: `invalid_json`, `invalid_body`, `invalid_request`
- 422: `validation_error`, `conflicting_id` (names the ID), `empty_input`
- 503: `model_not_loaded`

The model and config are loaded from `artifacts/` at startup, so a restart is safe (tested), and no training happens in a request. Each analysis is saved to `results/api_runs/<snapshot>__<input_hash>.json`.

`input_hash` is computed over the canonical, deduplicated input. Reordering the records or adding identical duplicates gives the same hash and identical decisions (tested).

## Web interface

The UI is `http://localhost:8000`: a single static file with plain JS and SVG and no external dependencies. It lets a reviewer:
- pick a val/test scenario or a staged snapshot and run it;
- see a lat/lon plot with accepted links drawn dashed, glitches marked ✕, and the selected tracklet's candidates dotted;
- read a per-tracklet table with status, successor, top-3 candidates and reason, plus the label with ✓/✗ for supplied datasets;
- use a **baseline / learned / compare** toggle, where compare highlights disagreements;
- run **Compare stages**, which shows only changed rows, both manifests, and stage-1 links in grey over stage 2;
- open a detail pane with summary measurements, the full candidate table and the original reports;
- see health, loading, empty, success and error states. "Send invalid sample" demonstrates the error state.

The UI was written by a Sonnet subagent against the API contract in PLAN.md §9, then reviewed by hand against real responses. **I verified the API calls it makes, but have not clicked through it in a browser in this session.**

## Demonstration checklist

| Requirement | Where |
|---|---|
| Clear continuation | staged `trk-aab05f6f4f → trk-6b3771dd35` (error 0.08 km, runner-up 0.007), or any accepted row in the UI |
| Competing plausible continuations | stage 1 `trk-4e94fbf3f4` (0.401 vs 0.401); failure case 1 (parallel aircraft) |
| A tracklet left unresolved | stage 1 baseline (33 unresolved, with reasons); failure cases 3 and 5 |
| Decision reassessed after more evidence | `results/staged_diff.json`, two "replaced" links; UI "Compare stages" |
| Identical results after reordering and duplicates | `tests/test_pipeline.py::test_reorder_and_duplicates_give_identical_results` (same `input_hash` and decisions) |

## Tests (`pytest -q`, 77 tests)

| File | Covers |
|---|---|
| `test_load.py` | required fields, empty and case-sensitive IDs, coordinate ranges, non-finite numbers and booleans, accepted and rejected timestamp formats, identical duplicates (key order, 1 vs 1.0), conflicts (null vs missing, array order, true vs 1, same instant written differently), ordering by `observed_at`, missing altitude, raw preservation, supporting-stream joins and conflicts |
| `test_decide.py` | one successor/predecessor, ambiguity on both sides, taken competitors, threshold, skip check, ranked measurements (NaN → null), strict time order in candidate generation, constraint checker (multi-predecessor, cycle) |
| `test_pipeline.py` | reorder + duplicate invariance, model save/load consistency, IDs are not features, missing altitude end to end, optional supporting streams, scenario isolation |
| `test_api.py` | health ok / 503, restart reload, response shape, every error code, malformed JSON, dataset endpoints |

## What the synthetic data simplifies, and what to validate with real data

The synthetic data makes several things easier than reality:
- Motion is a piecewise constant-heading/speed model with gentle turns. There are no holding patterns, approaches, wind, or dense terminal traffic.
- Noise is Gaussian and source-specific; there are no multipath or biased sensors.
- Each tracklet comes from one source, and tracklet boundaries are perfect.
- Gaps follow a simple mixture, and identifier errors are independent and random.
- **I wrote the generator myself**, so the field "investigation" could not be fully blind. I kept the source notes partial and based every interpretation on measured distributions, but a reviewer should weigh this.
- The learned model was trained on the generator's own manoeuvre distribution, so its advantage on large turns may not transfer.

With real observations I would validate:
- gap and turn-in-gap distributions;
- identifier reliability per source (callsign changes, code reuse);
- the `geo`/`baro` offset by region;
- score calibration before showing any "confidence";
- precision on dense parallel traffic such as arrival streams;
- the effect of the 0.02 learned threshold, where the test errors concentrated.

## Development history (disclosed, all before test)

1. **v0 baseline: raw km error.** Val recall stalled at 0.68 because long gaps were punished. I switched to error per gap-minute, and validation now chooses between the two variants.
2. **Per-minute baseline: skip errors.** It produced wrong links that were all *skips* (`scripts/diagnose.py`). I added the immediacy (skip) check to the shared decision rule, as a switch that validation selects.
3. **Selection rule:** changed from "max recall at precision ≥ 0.98" to an explicit cost (correct − 5 × wrong), so the cost of wrong links shapes the choice.

Test was run **once**, after these changes and the freeze. Nothing was changed after seeing the test results. The failure analysis and docs were written afterwards.

## Limitations and unfinished work

- Synthetic data only. The real data package was not available.
- Greedy association, not a global assignment. There is no turn-aware motion model (for example, constant-turn-rate projection), which would most help the baseline.
- The learned operating point sits at the edge of the validation grid, and its scores are uncalibrated.
- `q` is not used for noise-aware scoring, and the reported-speed fields are explored but unused. Both are possible next steps.
- No incremental updates: each snapshot is recomputed in full, which the brief allows.
- The UI was not exercised in a real browser during this session.

## Time spent

Approximate active time: _to be filled in by the candidate_. The AI-assisted build session covered planning, data generation, the pipeline, evaluation, API, UI, tests and docs.

## AI assistance

Claude Code (Claude Opus 5.5) planned the work and wrote the code and docs. A Claude Sonnet 5.5 subagent wrote the first version of the UI. Every prompt is logged in [PROMPTS.md](PROMPTS.md).

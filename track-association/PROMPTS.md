# AI Prompt Log

Tool: Claude Code (VS Code extension), model Claude Opus 5.5 (`claude-opus-5-5`).
Every prompt sent to the AI during this assessment is recorded below in order, followed by a short note on how the output was used.

---

## Prompt 1 — 2026-11-03

> Read the question, in @original_question.md , its cleaned out instruction using claude in @Aircraft_docs.md and the supporting docs in @Required_files/ .
>
> Makre a proper plan for its execution first, then implement it. Use suitable models for planning and coding. Also make sure you are logging the prompts and folloup prompts in a md file.

**How the output was used:** The AI read all three sources and found that they describe two different challenges:
- `original_question.md` / `Aircraft_docs.md` describe an ML **Aircraft Track Association** challenge (tracklets, train/val/test partitions, labels, staged snapshots).
- `Required_files/` (the three .docx files and `candidate-packet/`) describe a full-stack **Aircraft Movement History** challenge (POST /observations, POST /identity-mappings, GET /aircraft/{id}/history, replay.py, expected-statuses.json, concurrent-cases.json). The packet holds no tracklets, partitions or labels.

The AI stopped before planning and asked which challenge to build.

---

## Prompt 2 — follow-up (answer to the AI's clarification question)

> So do this thing, if it not mathching then work on the basis of making some assumptions and then proceed, use those files when its required later in the process.

**How the output was used:** Decision: build the **Track Association** challenge from `original_question.md`. Because its data package is missing, the AI writes a seeded synthetic data generator that produces the described package (partitions, supporting streams, labels, staged snapshots, source notes). Every assumption is documented in `PLAN.md` and `README.md`. Rules from `Required_files/Clarifications.docx` (UTC timestamp format, JSON-equality rules for duplicates) are reused in validation.

The AI then:
- wrote `PLAN.md`, planning on Opus;
- implemented the generator, pipeline, training, evaluation, staged demo, API and tests, coding on Opus;
- delegated the web UI to a Sonnet subagent (prompt below);
- reviewed and lightly edited the UI.

---

## Prompt 3 — follow-up

> Create a .gitignore file and update it

**How the output was used:** The AI added `track-association/.gitignore`. It ignores:
- the virtualenv and Python/pytest caches;
- the generated `data/` package, which is reproducible from seed 42 (`data/README_source_notes.md` stays tracked);
- per-request API logs in `results/api_runs/`;
- editor and OS files.

It keeps `artifacts/` and `results/`, because the brief requires saved models and results. It also added a note on regenerating the data to the README.

---

## Prompts sent by the AI to sub-agents during this session

### Sub-agent 1 — Web UI (model: Claude Sonnet 5.5, general-purpose agent)

> Write ONE file: `track-association/api/static/index.html` (create the folder if needed). Do not touch any other file. It is a small reviewer UI for an aircraft "track association" service, served by FastAPI from the same origin. Plain HTML + vanilla JS + inline SVG, NO external libraries/CDNs, works offline. Functional and clear, modest styling (light theme fine; include basic dark-mode via prefers-color-scheme with CSS variables). Must work at narrow widths without horizontal page scroll (tables can scroll inside a wrapper).
>
> Backend endpoints (same origin): GET /health -> 200 {"status":"ok","model_version":...} or 503; GET /datasets -> [{"name","label","kind"}] (staged names are "staged/stage_1" and "staged/stage_2"); GET /datasets/{name} -> request payload (pass straight through as POST body); POST /tracklets/associate -> RESPONSE or error {"error":{"code","message","details"}} with 400/422/503.
>
> RESPONSE shape: manifest {snapshot_id, input_hash, model_version, config_hash, decision_config, generated_at, n_observations_in, n_observations_unique, n_tracklets}; tracklets {tid: {scenario_id, source, n_points, start_time, end_time, points[{lat, lon, alt_ft, observed_at, observation_id, glitch}], raw[...], summary{...}}}; methods {baseline: {tid: RESULT}, learned: {tid: RESULT}}; metrics null or {baseline, learned}. RESULT = {status, successor, predecessor, score, reason, ranked_candidates[{candidate, score, rank, measurements}]}. Measurement keys include gap_min, dist_km, implied_speed_kmh, fwd_err_km, back_err_km, speed_diff_kmh, heading_change_deg, alt_diff_ft, callsign_match, cat_match — render generically; null shows as "—". Response may contain "labels" {tid: next|null}: if present show a truth column and mark ✓/✗ (evaluation display only).
>
> Required features: 1) header health status; 2) dataset select from /datasets with loading/error states, Run button; 3) method toggle baseline|learned|compare; 4) lat/lon SVG plot (cos(meanLat) aspect, ticks), polylines per tracklet with stable hue, start dot, end marker, tid label, accepted links dashed (both methods in compare mode, legend), glitch x markers, selected tracklet highlighted with faint dotted candidate links; 5) results table (tid, source, n_points, times, status badge, successor, top-3 candidates, reason; compare mode side-by-side with disagreements highlighted; row click selects); 6) detail pane (summary, full candidate table with measurements, collapsible raw reports JSON); 7) "Compare stages" button: fetch+POST staged/stage_1 and stage_2, diff table per method with change kind (new link, withdrawn, replaced, unchanged, new tracklet) and stage-2 reason, manifests of both, plot stage 2 with stage-1 links grey dashed; 8) manifest box + metrics table; 9) states: loading (disable buttons, "Running…"), empty, success, error (code + message + details) and a "Send invalid sample" button. Escape all strings inserted into HTML. Keep it under ~700 lines. Use `<title>Track Association</title>`. Reply with a short summary.

**How the output was used:** The file was kept. The main session reviewed it against real API responses and changed one thing: the stage-comparison table now shows only changed rows plus a count of unchanged ones.

# Data package — source notes

> **Provenance (assumption A1).** The challenge's data package was not in the supplied files. This package is produced by `python -m scripts.generate_data` (seed 42).
>
> The notes below are written the way a data provider would write them: core fields are fully documented, and optional fields only **partly** documented. What we later inferred about the optional fields is in the main README ("Field interpretation"). It is kept separate from these notes on purpose.

## Files (per partition: `train/`, `val/`, `test/`, `staged/snapshot_1/`, `staged/snapshot_2/`)

| File | One record = | Key (dedupe) | Joins to |
|---|---|---|---|
| `observations.jsonl` | one position report | `observation_id` | `tracklet_id` groups reports |
| `tracklet_context.jsonl` | context attached to a tracklet | `context_id` | `tracklet_id` |
| `id_reports.jsonl` | identity message attached to one report | `report_id` | `observation_id` |
| `tracklet_index.jsonl` | tracklet → scenario membership | `tracklet_id` | grouping only, not a feature |
| `labels.jsonl` | `{tracklet_id, next_tracklet_id or null}` | `tracklet_id` | **evaluation only** |

`staged/snapshots.json` gives the receipt cutoff of each snapshot. Each snapshot folder is a **complete** input that contains only records with `received_at <= cutoff`, and each has its own labels.

`_truth/` holds the generator's aircraft IDs. It is for debugging and failure analysis only, and no code under `src/` reads it.

## Core observation fields (documented)

`observation_id`, `tracklet_id`, `observed_at` (measurement time, UTC), `received_at` (platform receipt time, UTC), `latitude`, `longitude` (WGS84 degrees), `altitude_ft` (feet, may be null), `source`, `attributes` (source-specific object).

Timestamps use `Z` or `+00:00`. `feed-a` reports carry milliseconds.

Records may arrive duplicated, out of order, and late.

## Sources

- **feed-a**: high-rate receiver.
  - `gs`: ground speed.
  - `trk`: track angle.
  - `sq`: a code broadcast by the aircraft.
  - `rssi`: signal strength.
- **feed-b**: aggregator.
  - `spd`: speed (units not stated by the provider).
  - `c2`: (undocumented).
  - `ref`: an identifier assigned upstream.
  - `q`: quality class (meaning of the levels not stated).
  - `node`: internal ingest host.
- **feed-c**: low-rate secondary sensor.
  - `spd`: speed.
  - `hdg`: direction.
  - `vr`: (undocumented, sometimes absent).
  - `alt_ref`: altitude reference tag (`baro` or `geo`).

## Supporting streams

- `tracklet_context.cat`: an emitter category code (A1–A5). It may be missing, or it may disagree between records for the same tracklet.
- `tracklet_context.site`, `fw`: provider-internal.
- `id_reports.cs`: a callsign reported for that observation. Coverage is partial. Callsigns are not authoritative and can change or be wrong.
- `id_reports.src`: provider-internal.

## Simplified movement assumptions

- Aircraft fly at 280–520 kt ground speed (never more than ~600 kt).
- Turns are at most 2–3 °/s; between manoeuvres flight is straight and level or a steady climb/descent (≤ 2,500 ft/min).
- A straight-line, constant-velocity extrapolation is reasonable over a few minutes but degrades with longer gaps, because aircraft may turn while unobserved.
- Gaps between tracklets of one aircraft are usually under 30 min, rarely up to ~60 min.
- Position noise depends on the source; occasional isolated position glitches occur.

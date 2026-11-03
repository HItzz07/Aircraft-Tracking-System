"""Validation, canonical-JSON deduplication, grouping and supporting-stream joins.

Rules (brief + Required_files/Clarifications.docx, reused per PLAN.md A3):
* required: observation_id, tracklet_id, observed_at, received_at, latitude, longitude, source
* IDs/source: non-empty strings, case-sensitive, never trimmed
* timestamps: RFC3339 UTC ending in Z or +00:00, optional fraction of <= 3 digits;
  other offsets, naive times, invalid dates and leap seconds are rejected
* lat in [-90, 90], lon in [-180, 180], finite JSON numbers (not bool / str)
* altitude_ft: number or null (null is kept as None, never 0)
* attributes: optional, but if present must be an object (preserved verbatim)
* identical duplicates (same id, equal canonical JSON) are collapsed;
  same id + different content -> ValidationError naming the id
"""
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone

REQUIRED = ["observation_id", "tracklet_id", "observed_at", "received_at", "latitude", "longitude", "source"]
TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d{1,3})?(Z|\+00:00)$")


class ValidationError(Exception):
    def __init__(self, message, observation_id=None, code="validation_error", details=None):
        super().__init__(message)
        self.observation_id = observation_id
        self.code = code
        self.details = details or {}
        if observation_id is not None:
            self.details.setdefault("id", observation_id)


# ------------------------------------------------------------------ canonical JSON
def canonical(v):
    """Hashable canonical form: ignores key order, keeps array order, keeps JSON types
    distinct (true != 1, "1" != 1), treats 1 == 1.0, distinguishes missing from null."""
    if v is None:
        return ("null",)
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, str):
        return ("str", v)
    if isinstance(v, list):
        return ("arr", tuple(canonical(x) for x in v))
    if isinstance(v, dict):
        return ("obj", tuple(sorted((k, canonical(x)) for k, x in v.items())))
    raise ValidationError(f"unsupported JSON value type: {type(v).__name__}")


def canonical_json(v):
    """Stable text form used for hashing input snapshots."""
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ------------------------------------------------------------------ IO
def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValidationError(f"{path}:{line_no} is not valid JSON ({e.msg})", code="invalid_json")
    return records


# ------------------------------------------------------------------ field checks
def parse_utc(value, field, rid=None):
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string timestamp", rid)
    m = TS_RE.match(value)
    if not m:
        raise ValidationError(f"{field} must be RFC3339 UTC (Z or +00:00, <=3 fractional digits): {value!r}", rid)
    y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
    frac = m.group(7)
    micro = int((frac[1:] + "000")[:3]) * 1000 if frac else 0
    if s == 60:
        raise ValidationError(f"{field} leap seconds are not supported: {value!r}", rid)
    try:
        return datetime(y, mo, d, h, mi, s, micro, tzinfo=timezone.utc)
    except ValueError:
        raise ValidationError(f"{field} is not a valid calendar date/time: {value!r}", rid)


def is_finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def nonempty_str(rec, field, rid, required=True):
    v = rec.get(field)
    if v is None and not required:
        return None
    if not isinstance(v, str) or v == "":
        raise ValidationError(f"{field} must be a non-empty string", rid)
    return v


def validate_observation(rec):
    """Return a parsed copy of one raw record. The raw record is never modified."""
    if not isinstance(rec, dict):
        raise ValidationError("each observation must be a JSON object")
    rid = rec.get("observation_id") if isinstance(rec.get("observation_id"), str) else None
    for field in REQUIRED:
        if field not in rec or rec[field] is None:
            raise ValidationError(f"missing required field: {field}", rid)
    for field in ("observation_id", "tracklet_id", "source"):
        nonempty_str(rec, field, rid)
    lat, lon = rec["latitude"], rec["longitude"]
    if not is_finite_number(lat) or not -90 <= lat <= 90:
        raise ValidationError(f"invalid latitude: {lat!r}", rid)
    if not is_finite_number(lon) or not -180 <= lon <= 180:
        raise ValidationError(f"invalid longitude: {lon!r}", rid)
    alt = rec.get("altitude_ft")
    if alt is not None and not is_finite_number(alt):
        raise ValidationError("altitude_ft must be a finite number or null", rid)
    attrs = rec.get("attributes")
    if "attributes" in rec and not isinstance(attrs, dict):
        raise ValidationError("attributes must be an object when present", rid)
    observed = parse_utc(rec["observed_at"], "observed_at", rid)
    received = parse_utc(rec["received_at"], "received_at", rid)
    return {
        "observation_id": rec["observation_id"],
        "tracklet_id": rec["tracklet_id"],
        "observed_at": observed,          # movement continuity uses this
        "received_at": received,          # evidence availability only
        "lat": float(lat),
        "lon": float(lon),
        "alt_ft": None if alt is None else float(alt),
        "source": rec["source"],
        "attributes": attrs if isinstance(attrs, dict) else {},
        "raw": rec,
        "late_receipt_s": (received - observed).total_seconds(),
    }


# ------------------------------------------------------------------ dedupe
def dedupe(records, id_field, kind="observation"):
    """Collapse identical duplicates by id; reject same id with different content."""
    seen, order = {}, []
    for rec in records:
        if not isinstance(rec, dict):
            raise ValidationError(f"each {kind} record must be a JSON object")
        rid = rec.get(id_field)
        if not isinstance(rid, str) or rid == "":
            raise ValidationError(f"{id_field} must be a non-empty string", None)
        c = canonical(rec)
        if rid in seen:
            if seen[rid][0] != c:
                raise ValidationError(f"conflicting content for {id_field} {rid}", rid, code="conflicting_id",
                                      details={"id_field": id_field, "kind": kind})
            continue
        seen[rid] = (c, rec)
        order.append(rid)
    # deterministic order independent of input order
    return [seen[k][1] for k in sorted(order)]


def load_observations(raw_records):
    """Validate + dedupe + group into tracklets sorted by (observed_at, observation_id).

    Returns (tracklets: {tid: [obs...]}, stats)."""
    if not isinstance(raw_records, list):
        raise ValidationError("observations must be a list")
    unique = dedupe(raw_records, "observation_id")
    clean = [validate_observation(r) for r in unique]
    tracklets = defaultdict(list)
    for o in clean:
        tracklets[o["tracklet_id"]].append(o)
    for tid in tracklets:
        tracklets[tid].sort(key=lambda o: (o["observed_at"], o["observation_id"]))
        sources = {o["source"] for o in tracklets[tid]}
        if len(sources) > 1:
            # not forbidden by the brief; we keep it and note it in the summary
            pass
    stats = {"n_observations_in": len(raw_records), "n_observations_unique": len(unique),
             "n_duplicates_removed": len(raw_records) - len(unique), "n_tracklets": len(tracklets)}
    return {k: tracklets[k] for k in sorted(tracklets)}, stats


# ------------------------------------------------------------------ supporting streams
def load_supporting(supporting, tracklets):
    """Join supporting streams by their documented keys.

    tracklet_context: key context_id (dedupe), joined on tracklet_id
    id_reports:       key report_id (dedupe), joined on observation_id
    Records whose join key is not present in this snapshot are ignored (counted)."""
    supporting = supporting or {}
    if not isinstance(supporting, dict):
        raise ValidationError("supporting must be an object")
    obs_to_tid = {o["observation_id"]: tid for tid, obs in tracklets.items() for o in obs}
    ctx_by_t, cs_by_t = defaultdict(list), defaultdict(list)
    stats = {"tracklet_context_unmatched": 0, "id_reports_unmatched": 0}

    ctx = dedupe(supporting.get("tracklet_context") or [], "context_id", "tracklet_context")
    for c in ctx:
        tid = c.get("tracklet_id")
        if not isinstance(tid, str) or tid == "":
            raise ValidationError("tracklet_context.tracklet_id must be a non-empty string", c.get("context_id"))
        if tid not in tracklets:
            stats["tracklet_context_unmatched"] += 1
            continue
        ctx_by_t[tid].append(c)

    idr = dedupe(supporting.get("id_reports") or [], "report_id", "id_reports")
    for r in idr:
        oid = r.get("observation_id")
        if not isinstance(oid, str) or oid == "":
            raise ValidationError("id_reports.observation_id must be a non-empty string", r.get("report_id"))
        if oid not in obs_to_tid:
            stats["id_reports_unmatched"] += 1
            continue
        cs_by_t[obs_to_tid[oid]].append(r)
    stats["n_tracklet_context"] = len(ctx)
    stats["n_id_reports"] = len(idr)
    return dict(ctx_by_t), dict(cs_by_t), stats

"""Tracklet summaries and pair features.

Everything is computed from documented core fields (time, position, altitude) plus the
supporting signals justified in README "Field interpretation". IDs are never features.
Missing values are NaN / None, never 0.
"""
import math
from collections import Counter

import numpy as np

from src.geo import angle_diff, bearing_deg, from_local_km, haversine_km, heading_from_velocity, project, to_local_km

VEL_WINDOW_S = 180         # fit velocity on the last/first 3 minutes of a tracklet
MIN_FIT_POINTS = 3
GLITCH_SPEED_KMH = 1500    # movement assumption: groundspeed never exceeds ~600 kt (1111 km/h)
NAN = float("nan")

# Source-specific speed parsing (see README field table; units inferred from data, not names)
SPEED_TO_KMH = {("feed-a", "gs"): 1.852, ("feed-b", "spd"): 1.852, ("feed-c", "spd"): 3.6}

PHYSICS_FEATURES = ["gap_min", "dist_km", "implied_speed_kmh", "fwd_err_km", "back_err_km",
                    "speed_diff_kmh", "heading_change_deg", "bearing_align_deg", "speed_ratio", "alt_diff_ft"]
SUPPORT_FEATURES = ["callsign_match", "cat_match", "ref_match"]
ALL_FEATURES = PHYSICS_FEATURES + SUPPORT_FEATURES


def _speed_kmh(o1, o2):
    dt = (o2["observed_at"] - o1["observed_at"]).total_seconds()
    if dt <= 0:
        return math.inf
    return haversine_km(o1["lat"], o1["lon"], o2["lat"], o2["lon"]) / (dt / 3600)


def flag_glitches(obs):
    """A point is a glitch when reaching it AND leaving it both need > GLITCH_SPEED_KMH.
    Glitches are excluded from fits but preserved (and shown) in outputs."""
    n = len(obs)
    flags = [False] * n
    if n < 3:
        return flags
    for i in range(n):
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < n] if 0 < i < n - 1 else ([1, 2] if i == 0 else [n - 2, n - 3])
        if all(_speed_kmh(*sorted((obs[i], obs[j]), key=lambda o: o["observed_at"])) > GLITCH_SPEED_KMH for j in nbrs):
            flags[i] = True
    return flags


def _fit(points, at_end):
    """Least-squares constant-velocity fit over a window at one end.

    Returns (lat, lon, v_north_kmh, v_east_kmh, n_used) evaluated at the end time,
    or None when fewer than 2 usable points exist."""
    if len(points) < 2:
        return None
    t_ref = points[-1]["observed_at"] if at_end else points[0]["observed_at"]
    window = [p for p in points if abs((p["observed_at"] - t_ref).total_seconds()) <= VEL_WINDOW_S]
    if len(window) < MIN_FIT_POINTS:
        window = points[-MIN_FIT_POINTS:] if at_end else points[:MIN_FIT_POINTS]
    ref = window[-1] if at_end else window[0]
    t = np.array([(p["observed_at"] - t_ref).total_seconds() / 3600 for p in window])
    ne = np.array([to_local_km(ref["lat"], ref["lon"], p["lat"], p["lon"]) for p in window])
    if np.ptp(t) <= 0:
        return None
    A = np.vstack([np.ones_like(t), t]).T
    coef_n, *_ = np.linalg.lstsq(A, ne[:, 0], rcond=None)
    coef_e, *_ = np.linalg.lstsq(A, ne[:, 1], rcond=None)
    lat, lon = from_local_km(ref["lat"], ref["lon"], coef_n[0], coef_e[0])
    resid = ne - A @ np.vstack([coef_n, coef_e]).T
    rms_km = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
    return lat, lon, float(coef_n[1]), float(coef_e[1]), len(window), rms_km


def comparable_alt(p):
    """Altitude usable for cross-tracklet comparison, else None.

    feed-c records tagged alt_ref == "geo" sit ~+530 ft above every other source on true
    links (results/field_exploration.json); we interpret this as a different vertical
    reference and EXCLUDE them rather than apply an inferred correction."""
    if p["alt_ft"] is None:
        return None
    if p["source"] == "feed-c" and p["attributes"].get("alt_ref") == "geo":
        return None
    return p["alt_ft"]


def _alt(points, at_end, k=5):
    seq = list(reversed(points)) if at_end else points
    vals = [comparable_alt(p) for p in seq]
    vals = [v for v in vals if v is not None][:k]
    return float(np.median(vals)) if vals else None


def _reported_speed(points, at_end, k=5):
    seq = list(reversed(points)) if at_end else points
    vals = []
    for p in seq:
        for (src, key), f in SPEED_TO_KMH.items():
            v = p["attributes"].get(key)
            if p["source"] == src and isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(v * f)
        if len(vals) >= k:
            break
    return float(np.median(vals)) if vals else None


def _majority(values):
    """Majority value and whether a material conflict exists (>20 % disagreeing)."""
    vals = [v for v in values if isinstance(v, str) and v != ""]
    if not vals:
        return None, False
    c = Counter(vals).most_common()
    top, n = c[0]
    conflict = (len(vals) - n) / len(vals) > 0.2
    return (None if conflict else top), conflict


def summarize_tracklet(tid, obs, contexts=None, id_reports=None, scenario_id=None):
    glitch = flag_glitches(obs)
    good = [o for o, g in zip(obs, glitch) if not g] or obs
    end = _fit(good, at_end=True)
    start = _fit(good, at_end=False)
    first, last = good[0], good[-1]

    def unpack(fit, fallback):
        if fit is None:
            return fallback["lat"], fallback["lon"], None, None, None, None, 0, None
        lat, lon, vn, ve, n_used, rms = fit
        spd = math.hypot(vn, ve)
        return lat, lon, vn, ve, spd, heading_from_velocity(vn, ve), n_used, rms

    e_lat, e_lon, e_vn, e_ve, e_spd, e_hdg, e_n, e_rms = unpack(end, last)
    s_lat, s_lon, s_vn, s_ve, s_spd, s_hdg, s_n, s_rms = unpack(start, first)

    cs, cs_conflict = _majority([r.get("cs") for r in (id_reports or [])])
    cat, cat_conflict = _majority([c.get("cat") for c in (contexts or [])])
    refs = [o["attributes"].get("ref") for o in obs if o["source"] == "feed-b"]
    ref, ref_conflict = _majority(refs)
    dts = [(b["observed_at"] - a["observed_at"]).total_seconds() for a, b in zip(obs, obs[1:])]
    return {
        "tracklet_id": tid,
        "scenario_id": scenario_id,
        "source": "+".join(sorted({o["source"] for o in obs})),
        "n_points": len(obs),
        "n_glitches": int(sum(glitch)),
        "glitch_flags": glitch,
        "start_time": first["observed_at"], "end_time": last["observed_at"],
        "duration_s": (last["observed_at"] - first["observed_at"]).total_seconds(),
        "median_sample_interval_s": float(np.median(dts)) if dts else None,
        "start_lat": s_lat, "start_lon": s_lon, "end_lat": e_lat, "end_lon": e_lon,
        "end_vn": e_vn, "end_ve": e_ve, "end_speed_kmh": e_spd, "end_heading": e_hdg,
        "start_vn": s_vn, "start_ve": s_ve, "start_speed_kmh": s_spd, "start_heading": s_hdg,
        "end_fit_points": e_n, "end_fit_rms_km": e_rms, "start_fit_points": s_n, "start_fit_rms_km": s_rms,
        "end_alt_ft": _alt(good, True), "start_alt_ft": _alt(good, False),
        "alt_missing_fraction": sum(o["alt_ft"] is None for o in obs) / len(obs),
        "end_reported_speed_kmh": _reported_speed(good, True),
        "start_reported_speed_kmh": _reported_speed(good, False),
        "callsign": cs, "callsign_conflict": cs_conflict, "n_id_reports": len(id_reports or []),
        "cat": cat, "cat_conflict": cat_conflict, "n_context": len(contexts or []),
        "ref": ref if "feed-b" in {o["source"] for o in obs} else None, "ref_conflict": ref_conflict,
        "max_receipt_delay_s": max(o["late_receipt_s"] for o in obs),
    }


def _match(a, b):
    if a is None or b is None:
        return NAN
    return 1.0 if a == b else 0.0


def pair_features(a, b):
    gap_h = (b["start_time"] - a["end_time"]).total_seconds() / 3600
    dist = haversine_km(a["end_lat"], a["end_lon"], b["start_lat"], b["start_lon"])
    implied = dist / gap_h if gap_h > 0 else math.inf

    if a["end_vn"] is not None:
        plat, plon = project(a["end_lat"], a["end_lon"], a["end_vn"], a["end_ve"], gap_h)
        fwd = haversine_km(plat, plon, b["start_lat"], b["start_lon"])
    else:
        fwd = dist
    if b["start_vn"] is not None:
        plat, plon = project(b["start_lat"], b["start_lon"], b["start_vn"], b["start_ve"], -gap_h)
        back = haversine_km(plat, plon, a["end_lat"], a["end_lon"])
    else:
        back = dist

    sa, sb = a["end_speed_kmh"], b["start_speed_kmh"]
    speed_diff = abs(sa - sb) if sa is not None and sb is not None else NAN
    speed_ratio = implied / ((sa + sb) / 2) if sa and sb else NAN
    hdg_change = angle_diff(a["end_heading"], b["start_heading"]) if a["end_heading"] is not None and b["start_heading"] is not None else NAN
    align = (angle_diff(a["end_heading"], bearing_deg(a["end_lat"], a["end_lon"], b["start_lat"], b["start_lon"]))
             if a["end_heading"] is not None and dist > 0.5 else NAN)
    alt_diff = abs(a["end_alt_ft"] - b["start_alt_ft"]) if a["end_alt_ft"] is not None and b["start_alt_ft"] is not None else NAN
    return {
        "gap_min": gap_h * 60,
        "dist_km": dist,
        "implied_speed_kmh": implied,
        "fwd_err_km": fwd,
        "back_err_km": back,
        "speed_diff_kmh": speed_diff,
        "heading_change_deg": hdg_change,
        "bearing_align_deg": align,
        "speed_ratio": speed_ratio,
        "alt_diff_ft": alt_diff,
        "callsign_match": _match(a["callsign"], b["callsign"]),
        "cat_match": _match(a["cat"], b["cat"]),
        "ref_match": _match(a["ref"], b["ref"]),
    }

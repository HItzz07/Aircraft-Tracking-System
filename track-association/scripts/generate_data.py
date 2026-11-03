#!/usr/bin/env python3
"""Seeded synthetic data generator for the track-association exercise.

ASSUMPTION A1 (see PLAN.md): the challenge's data package was not supplied, so this
script produces an equivalent package. Ground truth (aircraft identity) is written
ONLY to labels.jsonl and data/_truth/. Nothing under src/ reads _truth/.

    python -m scripts.generate_data          # writes ./data
"""
import json
import math
import random
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 42
R_KM = 6371.0
KT_KMH = 1.852
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"

SOURCES = ["feed-a", "feed-b", "feed-c"]
SOURCE_W = [0.40, 0.35, 0.25]
SAMPLING_S = {"feed-a": (4, 12), "feed-b": (8, 25), "feed-c": (15, 40)}
NOISE_M = {"feed-a": 30.0, "feed-c": 250.0}
Q_NOISE_M = {"1": 60.0, "2": 120.0, "3": 300.0}
CATS = ["A1", "A2", "A3", "A4", "A5"]
CAT_W = [0.1, 0.25, 0.4, 0.15, 0.1]
AIRLINES = ["GVA", "NRT", "BLU", "SKY", "AXL", "KLM", "TRX", "OPL"]
SHARED_SQ = ["2000", "7000", "1200"]

PARTITIONS = {
    # name: (n_scenarios, base date)
    "train": (60, datetime(2026, 1, 1, 8, tzinfo=timezone.utc)),
    "val": (20, datetime(2026, 3, 1, 8, tzinfo=timezone.utc)),
    "test": (20, datetime(2026, 5, 1, 8, tzinfo=timezone.utc)),
}


# ---------------------------------------------------------------- helpers
def clip(x, lo, hi):
    return max(lo, min(hi, x))


def signed_diff(a, b):
    return ((a - b + 540.0) % 360.0) - 180.0


def offset(lat, lon, north_km, east_km):
    dlat = math.degrees(north_km / R_KM)
    dlon = math.degrees(east_km / (R_KM * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def dest(lat, lon, bearing, dist_km):
    b = math.radians(bearing)
    return offset(lat, lon, dist_km * math.cos(b), dist_km * math.sin(b))


def fmt_ts(dt, style):
    if style == "ms":
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    base = dt.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")
    return base + ("+00:00" if style == "offset" else "Z")


def hexid(rng, n):
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


# ---------------------------------------------------------------- motion
def simulate(rng, lat, lon, hdg, spd, alt, duration_s):
    """1 s steps. Returns list of (lat, lon, heading, speed_kt, alt_ft, vr_fpm)."""
    states = []
    target_hdg, turn_rate, target_spd = hdg, 0.0, spd
    target_alt, vr_mag = alt, 0.0
    next_m = rng.uniform(300, 1800)
    for t in range(duration_s + 1):
        dalt = target_alt - alt
        vr = 0.0 if abs(dalt) < 1 else math.copysign(vr_mag, dalt)
        states.append((lat, lon, hdg, spd, alt, vr))
        if t >= next_m:
            k = rng.random()
            if k < 0.6:
                target_hdg = (hdg + rng.choice([-1, 1]) * rng.uniform(15, 120)) % 360
                turn_rate = rng.uniform(0.5, 2.0)
            elif k < 0.8:
                target_spd = clip(spd + rng.uniform(-40, 40), 280, 520)
            else:
                target_alt = clip(alt + rng.choice([-1, 1]) * rng.uniform(2000, 8000), 8000, 41000)
                vr_mag = rng.uniform(800, 2500)
            next_m = t + rng.uniform(300, 1800)
        d = signed_diff(target_hdg, hdg)
        if abs(d) > 1e-9:
            hdg = (hdg + math.copysign(min(abs(d), turn_rate), d)) % 360
        spd += clip(target_spd - spd, -0.5, 0.5)
        if abs(dalt) >= 1:
            alt += math.copysign(min(vr_mag / 60.0, abs(dalt)), dalt)
        step = spd * KT_KMH / 3600.0
        lat, lon = dest(lat, lon, hdg, step)
    return states


def follower_states(leader, side, dist_km, alt_off):
    out = []
    for lat, lon, hdg, spd, alt, vr in leader:
        flat, flon = dest(lat, lon, hdg + 90 * side, dist_km)
        out.append((flat, flon, hdg, spd, alt + alt_off, vr))
    return out


def state_at(states, t):
    i = int(math.floor(t))
    i = clip(i, 0, len(states) - 2)
    f = clip(t - i, 0.0, 1.0)
    a, b = states[i], states[i + 1]
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2], a[3], a[4] + (b[4] - a[4]) * f, a[5])


# ---------------------------------------------------------------- scenario
def make_aircraft(rng, center, t_start, n_existing, leader=None):
    ident = {
        "cs": f"{rng.choice(AIRLINES)}{rng.randint(10, 9999)}",
        "icao": hexid(rng, 6).upper(),
        "sq": rng.choice(SHARED_SQ) if rng.random() < 0.1 else "".join(rng.choice("01234567") for _ in range(4)),
        "cat": rng.choices(CATS, CAT_W)[0],
    }
    if leader is not None:
        side = rng.choice([-1, 1])
        states = follower_states(leader["states"], side, rng.uniform(4, 12), rng.choice([0, 1000, -1000, 2000]))
        ident["cat"] = leader["ident"]["cat"] if rng.random() < 0.7 else ident["cat"]
        return {"states": states, "t_start": leader["t_start"] + rng.uniform(-90, 90), "ident": ident, "follower": True}
    ang, r = rng.uniform(0, 360), rng.uniform(40, 130)
    slat, slon = dest(center[0], center[1], ang, r)
    hdg = (ang + 180 + rng.uniform(-50, 50)) % 360
    duration = int(rng.uniform(70, 150) * 60)
    states = simulate(rng, slat, slon, hdg, rng.uniform(300, 500), float(rng.randrange(18000, 39001, 1000)), duration)
    return {"states": states, "t_start": t_start, "ident": ident, "follower": False}


def cut_tracklets(rng, duration):
    segs, t = [], rng.uniform(0, 120)
    while t < duration - 90:
        end = min(t + rng.uniform(3, 18) * 60, duration)
        if end - t >= 90:
            segs.append((t, end))
        gap = rng.uniform(60, 480) if rng.random() < 0.7 else rng.uniform(480, 1500)
        t = end + gap
    return segs


def gen_scenario(rng, scenario_id, t0):
    center = (rng.uniform(35, 55), rng.uniform(-10, 20))
    geo_off = rng.uniform(300, 700)
    n = rng.randint(6, 12)
    aircraft = []
    while len(aircraft) < n:
        ac = make_aircraft(rng, center, rng.uniform(0, 45 * 60), len(aircraft))
        aircraft.append(ac)
        if rng.random() < 0.3 and len(aircraft) < n:
            aircraft.append(make_aircraft(rng, center, None, len(aircraft), leader=ac))

    observations, id_reports, contexts, truth = [], [], [], []
    all_cs = [a["ident"]["cs"] for a in aircraft]
    all_icao = [a["ident"]["icao"] for a in aircraft]
    for ai, ac in enumerate(aircraft):
        aid = f"{scenario_id}-ac{ai:02d}"
        duration = len(ac["states"]) - 1
        segs = cut_tracklets(rng, duration)
        ident = dict(ac["ident"])
        cs_change_at = rng.randint(1, max(1, len(segs) - 1)) if rng.random() < 0.06 else None
        sq_change_at = rng.randint(1, max(1, len(segs) - 1)) if rng.random() < 0.15 else None
        for si, (ts, te) in enumerate(segs):
            if si > 0 and rng.random() < 0.10:
                continue  # dropped tracklet: never observed
            if si == cs_change_at:
                ident["cs"] = f"{rng.choice(AIRLINES)}{rng.randint(10, 9999)}"
            if si == sq_change_at:
                ident["sq"] = "".join(rng.choice("01234567") for _ in range(4))
            tid = f"trk-{hexid(rng, 10)}"
            source = rng.choices(SOURCES, SOURCE_W)[0]
            obs, recs = make_tracklet_obs(rng, tid, source, ac, ts, te, t0, ident, all_icao, geo_off)
            observations.extend(obs)
            truth.append({"tracklet_id": tid, "aircraft_id": aid, "scenario_id": scenario_id, "source": source,
                          "start": recs[0]["observed_at"], "end": recs[-1]["observed_at"], "segment_index": si,
                          "follower": ac["follower"]})
            id_reports.extend(make_id_reports(rng, obs, ident["cs"], all_cs))
            contexts.extend(make_contexts(rng, tid, obs, ident["cat"]))
    return observations, id_reports, contexts, truth


def make_tracklet_obs(rng, tid, source, ac, ts, te, t0, ident, all_icao, geo_off):
    lo, hi = SAMPLING_S[source]
    times, t = [], ts
    while t <= te:
        times.append(t)
        t += rng.uniform(lo, hi) + (rng.uniform(30, 90) if rng.random() < 0.03 else 0)
    while len(times) < 3:
        times.append(times[-1] + lo)
    q_t = rng.choices(["1", "2", "3"], [0.5, 0.35, 0.15])[0]
    alt_all_null = source == "feed-b" and rng.random() < 0.2
    alt_ref = "baro" if rng.random() < 0.6 else "geo"
    wrong_ref = rng.random() < 0.04
    tracklet_ref = rng.choice(all_icao) if wrong_ref else ident["icao"]
    node = f"ingest-{rng.randint(1, 24):02d}"
    ts_style = {"feed-a": "ms", "feed-b": "z", "feed-c": "offset"}[source]
    obs, recs = [], []
    for t in times:
        if source != "feed-a":
            t = round(t)
        lat, lon, hdg, spd, alt, vr = state_at(ac["states"], t)
        q = q_t if rng.random() < 0.9 else rng.choice(["1", "2", "3"])
        sigma = Q_NOISE_M[q] if source == "feed-b" else NOISE_M[source]
        nlat, nlon = offset(lat, lon, rng.gauss(0, sigma) / 1000, rng.gauss(0, sigma) / 1000)
        if rng.random() < 0.003:  # position glitch
            nlat, nlon = dest(nlat, nlon, rng.uniform(0, 360), rng.uniform(5, 30))
        observed = t0 + timedelta(seconds=ac["t_start"] + t)
        delay = rng.lognormvariate(1.0, 0.8)
        if source == "feed-b" and rng.random() < 0.04:
            delay += rng.uniform(120, 600)
        received = observed + timedelta(seconds=delay)
        if source == "feed-a":
            alt_v = None if rng.random() < 0.03 else round(alt / 25) * 25
            attrs = {"gs": round(spd + rng.gauss(0, 3)), "trk": round((hdg + rng.gauss(0, 1.5)) % 360, 1),
                     "sq": ident["sq"], "rssi": round(rng.uniform(-95, -60), 1)}
        elif source == "feed-b":
            alt_v = None if (alt_all_null or rng.random() < 0.1) else round(alt / 100) * 100
            attrs = {"spd": round(spd + rng.gauss(0, 6)), "c2": round((hdg + rng.gauss(0, 4)) % 360, 1)}
            u = rng.random()
            if u < 0.05:
                pass  # ref key absent
            elif u < 0.08:
                attrs["ref"] = None
            else:
                attrs["ref"] = tracklet_ref
            attrs["q"] = q
            attrs["node"] = node
        else:
            alt_v = None if rng.random() < 0.35 else round(alt + (geo_off if alt_ref == "geo" else 0))
            attrs = {"spd": round(spd * 0.514444 + rng.gauss(0, 2), 1), "hdg": round((hdg + rng.gauss(0, 3)) % 360)}
            if rng.random() > 0.2:
                attrs["vr"] = int(vr + rng.gauss(0, 150))
            attrs["alt_ref"] = alt_ref
        rec = {
            "observation_id": f"obs-{hexid(rng, 12)}",
            "tracklet_id": tid,
            "observed_at": fmt_ts(observed, ts_style),
            "received_at": fmt_ts(received, "ms" if ts_style == "ms" else "z"),
            "latitude": round(nlat, 6),
            "longitude": round(nlon, 6),
            "altitude_ft": alt_v,
            "source": source,
            "attributes": attrs,
        }
        obs.append(rec)
        recs.append(rec)
    return obs, recs


def make_id_reports(rng, obs, cs, all_cs):
    if rng.random() > 0.65:
        return []
    frac = rng.uniform(0.3, 1.0)
    tracklet_cs = rng.choice(all_cs) if rng.random() < 0.03 else cs
    out = []
    for o in obs:
        if rng.random() > frac:
            continue
        v = tracklet_cs
        u = rng.random()
        if u < 0.03:
            v = None
        elif u < 0.05:
            v = rng.choice(all_cs)  # conflicting value inside one tracklet
        rec = {"report_id": f"idr-{hexid(rng, 12)}", "observation_id": o["observation_id"],
               "received_at": o["received_at"], "cs": v, "src": rng.choice(["ident", "fpl", "ident"])}
        out.append(rec)
        if rng.random() < 0.01:
            out.append(json.loads(json.dumps(rec)))
    return out


def make_contexts(rng, tid, obs, cat):
    if rng.random() > 0.9:
        return []
    first_rx = min(o["received_at"] for o in obs)
    rx = (datetime.fromisoformat(first_rx.replace("Z", "+00:00")) + timedelta(seconds=rng.uniform(5, 60)))
    v = cat
    u = rng.random()
    if u < 0.05:
        v = None
    elif u < 0.10:
        v = rng.choice(CATS)
    rec = {"context_id": f"ctx-{hexid(rng, 10)}", "tracklet_id": tid, "received_at": fmt_ts(rx, "z"),
           "cat": v, "site": f"S-{rng.randint(1, 40):02d}", "fw": rng.choice(["3.1.4", "3.2.0", "2.9.9"])}
    out = [rec]
    if rng.random() < 0.05:
        out.append(dict(rec))  # redundant identical copy
    if rng.random() < 0.03:
        other = dict(rec, context_id=f"ctx-{hexid(rng, 10)}", cat=rng.choice(CATS))
        out.append(other)  # conflicting second record for same tracklet
    return out


# ---------------------------------------------------------------- packaging
def add_duplicates(rng, records, p=0.015):
    out = list(records)
    for r in records:
        if rng.random() < p:
            dup = json.loads(json.dumps(r))
            if rng.random() < 0.5:  # same content, different key order
                dup = {k: dup[k] for k in reversed(list(dup))}
                if isinstance(dup.get("attributes"), dict):
                    dup["attributes"] = {k: dup["attributes"][k] for k in reversed(list(dup["attributes"]))}
            out.append(dup)
    rng.shuffle(out)
    return out


def labels_from_truth(truth_rows):
    by_ac = {}
    for t in truth_rows:
        by_ac.setdefault(t["aircraft_id"], []).append(t)
    labels = []
    for rows in by_ac.values():
        rows.sort(key=lambda r: r["start"])
        for i, r in enumerate(rows):
            nxt = rows[i + 1]["tracklet_id"] if i + 1 < len(rows) else None
            labels.append({"tracklet_id": r["tracklet_id"], "next_tracklet_id": nxt, "scenario_id": r["scenario_id"]})
    labels.sort(key=lambda r: r["tracklet_id"])
    return labels


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_package(folder, obs, idr, ctx, truth, rng):
    write_jsonl(folder / "observations.jsonl", add_duplicates(rng, obs))
    idr = list(idr); rng.shuffle(idr)
    ctx = list(ctx); rng.shuffle(ctx)
    write_jsonl(folder / "id_reports.jsonl", idr)
    write_jsonl(folder / "tracklet_context.jsonl", ctx)
    index = sorted({(t["tracklet_id"], t["scenario_id"]) for t in truth})
    write_jsonl(folder / "tracklet_index.jsonl", [{"tracklet_id": a, "scenario_id": b} for a, b in index])
    write_jsonl(folder / "labels.jsonl", labels_from_truth(truth))


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build_staged(rng):
    """One scenario with receipt delays designed so that snapshot 1 misses some fragments."""
    t0 = datetime(2026, 7, 1, 8, tzinfo=timezone.utc)
    sid = "scn-staged-000"
    while True:
        obs, idr, ctx, truth = gen_scenario(rng, sid, t0)
        by_ac = {}
        for t in truth:
            by_ac.setdefault(t["aircraft_id"], []).append(t)
        multi = [sorted(v, key=lambda r: r["start"]) for v in by_ac.values() if len(v) >= 3]
        if len(multi) >= 3:
            break
    t1 = t0 + timedelta(hours=3, minutes=30)
    t2 = t1 + timedelta(hours=2)
    delayed_whole = {multi[0][1]["tracklet_id"], multi[1][1]["tracklet_id"]}   # middle fragments
    delayed_last = {multi[2][-1]["tracklet_id"]}                                # a final fragment
    tail_tid = multi[2][0]["tracklet_id"]                                       # tail of a fragment
    tail_obs = sorted([o for o in obs if o["tracklet_id"] == tail_tid], key=lambda o: o["observed_at"])
    tail_ids = {o["observation_id"] for o in tail_obs[int(len(tail_obs) * 0.6):]}
    late = {}
    for o in obs:
        if o["tracklet_id"] in delayed_whole | delayed_last or o["observation_id"] in tail_ids:
            rx = t1 + timedelta(minutes=rng.uniform(5, 60))
            o["received_at"] = fmt_ts(rx, "z")
            late[o["observation_id"]] = o["received_at"]
    for r in idr:
        if r["observation_id"] in late:
            r["received_at"] = late[r["observation_id"]]
    for c in ctx:
        if c["tracklet_id"] in delayed_whole | delayed_last:
            c["received_at"] = fmt_ts(t1 + timedelta(minutes=rng.uniform(5, 60)), "z")
    design = {"snapshot_1_cutoff": fmt_ts(t1, "z"), "snapshot_2_cutoff": fmt_ts(t2, "z"),
              "designed_delays": {"whole_middle_tracklets": sorted(delayed_whole),
                                  "whole_final_tracklets": sorted(delayed_last),
                                  "tail_delayed_tracklet": tail_tid}}
    for k, cutoff in [("snapshot_1", t1), ("snapshot_2", t2)]:
        o_k = [o for o in obs if parse(o["received_at"]) <= cutoff]
        present = {o["tracklet_id"] for o in o_k}
        i_k = [r for r in idr if parse(r["received_at"]) <= cutoff]
        c_k = [c for c in ctx if parse(c["received_at"]) <= cutoff and c["tracklet_id"] in present]
        tr_k = []
        for t in truth:
            mine = sorted(o["observed_at"] for o in o_k if o["tracklet_id"] == t["tracklet_id"])
            if mine:
                tr_k.append(dict(t, start=mine[0], end=mine[-1]))
        write_package(OUT / "staged" / k, o_k, i_k, c_k, tr_k, random.Random(f"{SEED}-pack-{k}"))
        write_jsonl(OUT / "_truth" / f"staged_{k}.jsonl", tr_k)
    (OUT / "staged" / "snapshots.json").write_text(json.dumps({
        "scenario_id": sid,
        "snapshots": [{"snapshot_id": "stage_1", "folder": "snapshot_1", "received_cutoff": design["snapshot_1_cutoff"]},
                      {"snapshot_id": "stage_2", "folder": "snapshot_2", "received_cutoff": design["snapshot_2_cutoff"]}],
        "note": "Each snapshot is a complete input containing only records with received_at <= its cutoff."
    }, indent=2), encoding="utf-8")
    (OUT / "_truth" / "staged_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")


def main():
    for sub in ["train", "val", "test", "staged", "_truth"]:
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
    for part, (n, base) in PARTITIONS.items():
        all_obs, all_idr, all_ctx, all_truth = [], [], [], []
        for i in range(n):
            rng = random.Random(f"{SEED}-{part}-{i}")
            sid = f"scn-{part}-{i:03d}"
            o, idr, c, tr = gen_scenario(rng, sid, base + timedelta(days=i))
            all_obs += o; all_idr += idr; all_ctx += c; all_truth += tr
        write_package(OUT / part, all_obs, all_idr, all_ctx, all_truth, random.Random(f"{SEED}-pack-{part}"))
        write_jsonl(OUT / "_truth" / f"{part}.jsonl", all_truth)
        print(f"{part}: {n} scenarios, {len(all_truth)} tracklets, {len(all_obs)} observations")
    build_staged(random.Random(f"{SEED}-staged"))
    print("staged: written")


if __name__ == "__main__":
    main()

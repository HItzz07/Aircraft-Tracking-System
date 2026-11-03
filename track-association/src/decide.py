"""Deterministic greedy association shared by both scorers.

Pairs are visited best-first (score desc, then ids for ties). For pair A -> B:
  1. A already has an accepted successor        -> skip
  2. score < threshold                          -> withhold ("too weak")
  3. B already has an accepted predecessor      -> withhold ("conflict")
  4. another still-available successor of A, or another still-available predecessor
     of B, scores within `margin`               -> withhold ("ambiguous")
  5. otherwise accept.
"Still available" = not already consumed by an accepted link, so a competitor that lost
its slot to a stronger link no longer blocks. Constraints:
  * one successor / one predecessor: enforced by steps 1 and 3
  * strict time order: candidates only contain B.start > A.end
  * no cycles: every link moves strictly forward in time, so a cycle is impossible
Immediacy ("skip") check, optional via `skip_check`:
  A -> C is withheld when some B exists with A -> B and B -> C both scoring >= threshold
  (B plausibly sits inside the A-C gap, so C would not be A's *immediate* successor).
  Such skip pairs also do not count as ambiguity competitors for A -> B or B -> C,
  because A -> B -> C and A -> C describe the same aircraft, not rival explanations.
Limitation: greedy is not globally optimal; a strong wrong link can block a correct one.
"""
import math
from collections import defaultdict

TOP_K = 5


def _clean(v, nd=3):
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return None
        return round(v, nd)
    return v


def _fmt(f):
    parts = [f"gap {f['gap_min']:.1f} min", f"mean projection error {(f['fwd_err_km'] + f['back_err_km']) / 2:.2f} km"]
    if f.get("heading_change_deg") is not None and not math.isnan(f["heading_change_deg"]):
        parts.append(f"heading change {f['heading_change_deg']:.0f} deg")
    return ", ".join(parts)


def find_skips(pairs, threshold):
    """{(a, c): b} for every candidate a->c that could pass through an intermediate b."""
    strong = defaultdict(list)
    score = {}
    for a, b, f, s in pairs:
        score[(a, b)] = (s, f["gap_min"])
        if s >= threshold:
            strong[a].append(b)
    skips = {}
    for a, bs in strong.items():
        for b in sorted(bs, key=lambda x: (score[(a, x)][1], x)):   # nearest intermediate first
            for c in strong.get(b, []):
                if (a, c) in score and (a, c) not in skips:
                    skips[(a, c)] = b
    return skips


def decide(tracklet_ids, candidates, scores, threshold, margin, top_k=TOP_K, skip_check=True):
    pairs = [(a, b, f, float(s)) for (a, b, f), s in zip(candidates, scores)]
    skips = find_skips(pairs, threshold) if skip_check else {}
    succ_opts, pred_opts = defaultdict(list), defaultdict(list)
    for a, b, _, s in pairs:
        if (a, b) in skips:
            continue  # a skip pair is not a rival explanation
        succ_opts[a].append((s, b))
        pred_opts[b].append((s, a))
    pairs.sort(key=lambda p: (-p[3], p[0], p[1]))

    successor, predecessor, reasons, acc_score = {}, {}, {}, {}
    for a, b, f, s in pairs:
        if a in successor:
            continue
        if s < threshold:
            reasons.setdefault(a, f"withheld: best remaining candidate {b} scores {s:.3f}, below threshold {threshold} ({_fmt(f)})")
            continue
        if (a, b) in skips:
            mid = skips[(a, b)]
            reasons.setdefault(a, f"withheld {b} ({s:.3f}): {mid} fits inside the gap ({a}->{mid} and {mid}->{b} "
                                  f"both >= threshold), so {b} is probably not the immediate successor")
            continue
        if b in predecessor:
            reasons.setdefault(a, f"withheld: candidate {b} ({s:.3f}) already accepted as successor of "
                                  f"{predecessor[b]} with a stronger score ({acc_score[predecessor[b]]:.3f})")
            continue
        alt_a = max(((s2, b2) for s2, b2 in succ_opts[a] if b2 != b and b2 not in predecessor), default=None)
        alt_b = max(((s2, a2) for s2, a2 in pred_opts[b] if a2 != a and a2 not in successor), default=None)
        if alt_a is not None and s - alt_a[0] < margin:
            reasons.setdefault(a, f"withheld (ambiguous): {b} ({s:.3f}) vs {alt_a[1]} ({alt_a[0]:.3f}); "
                                  f"difference {s - alt_a[0]:.3f} < margin {margin}")
            continue
        if alt_b is not None and s - alt_b[0] < margin:
            reasons.setdefault(a, f"withheld (ambiguous): {b} fits after {a} ({s:.3f}) almost as well as after "
                                  f"{alt_b[1]} ({alt_b[0]:.3f}); difference {s - alt_b[0]:.3f} < margin {margin}")
            continue
        successor[a], predecessor[b], acc_score[a] = b, a, s
        runner = f"; runner-up {alt_a[1]} ({alt_a[0]:.3f})" if alt_a else "; no other available successor"
        reasons[a] = f"accepted {b}: score {s:.3f} >= {threshold}, {_fmt(f)}{runner}"

    ranked = defaultdict(list)
    for a, b, f, s in sorted(pairs, key=lambda p: (p[0], -p[3], p[1])):
        ranked[a].append({"candidate": b, "score": round(s, 4), "rank": len(ranked[a]) + 1,
                          "measurements": {k: _clean(v) for k, v in f.items()}})

    results = {}
    for tid in sorted(tracklet_ids):
        succ = successor.get(tid)
        results[tid] = {
            "status": "accepted" if succ else "unresolved",
            "successor": succ,
            "predecessor": predecessor.get(tid),
            "score": round(acc_score[tid], 4) if succ else None,
            "n_candidates": len(ranked[tid]),
            "ranked_candidates": ranked[tid][:top_k],
            "reason": reasons.get(tid, "unresolved: no candidate passed candidate generation (time/position gates)"),
        }
    return results


def check_constraints(results, summaries_by_id):
    """Raise AssertionError if any association constraint is violated (used in tests + pipeline)."""
    preds = defaultdict(list)
    for a, r in results.items():
        b = r["successor"]
        if b is None:
            continue
        preds[b].append(a)
        assert summaries_by_id[b]["start_time"] > summaries_by_id[a]["end_time"], f"{a}->{b} not strictly after"
    for b, lst in preds.items():
        assert len(lst) == 1, f"{b} has multiple predecessors {lst}"
    for start in results:  # cycle check by walking chains
        seen, cur = set(), start
        while cur is not None:
            assert cur not in seen, f"cycle through {cur}"
            seen.add(cur)
            cur = results.get(cur, {}).get("successor")

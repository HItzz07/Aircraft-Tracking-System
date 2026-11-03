"""Metrics exactly as defined in the brief. Zero denominators -> None (reported as N/A)."""
from src.load import read_jsonl


def load_labels(path):
    return {r["tracklet_id"]: r.get("next_tracklet_id") for r in read_jsonl(path)}


def safe_div(n, d):
    return None if d == 0 else n / d


def metrics(results, labels, candidates):
    """results: {tid: result} over the WHOLE partition; labels: {tid: next|None};
    candidates: every generated (a, b, f) for the partition."""
    accepted = {a: r["successor"] for a, r in results.items() if r["successor"]}
    true_links = {a: b for a, b in labels.items() if b is not None}
    cand = {(a, b) for a, b, _ in candidates}
    correct = sum(1 for a, b in accepted.items() if labels.get(a) == b)
    wrong_on_no_successor = sum(1 for a in accepted if a in labels and labels[a] is None)
    unresolved = sum(1 for a in true_links if a not in accepted)
    kept = sum(1 for a, b in true_links.items() if (a, b) in cand)
    return {
        "tracklets": len(labels),
        "labeled_links": len(true_links),
        "accepted_links": len(accepted),
        "correct_links": correct,
        "incorrect_links": len(accepted) - correct,
        "incorrect_on_no_successor_tracklets": wrong_on_no_successor,
        "precision": safe_div(correct, len(accepted)),
        "recall": safe_div(correct, len(true_links)),
        "unresolved_with_labeled_successor": unresolved,
        "unresolved_proportion": safe_div(unresolved, len(true_links)),
        "candidate_links_kept": kept,
        "candidate_recall": safe_div(kept, len(true_links)),
        "n_candidate_pairs": len(cand),
    }

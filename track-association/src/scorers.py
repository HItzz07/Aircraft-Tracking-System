"""Pair scorers. Direction for both: HIGHER = more plausible continuation.

baseline: score = -((fwd_err_km + back_err_km) / 2) / (2 + gap_min)
          i.e. minus the mean straight-line prediction mismatch (A's end velocity
          projected forward, B's start velocity projected backward), expressed per
          minute of gap (+2 min to absorb position noise on very short gaps), because
          an aircraft can drift further from a straight line the longer it is unseen.
          Units km/min. 0 = perfect fit; -0.5 = ends miss by 0.5 km per gap-minute.
          A physical continuity score, NOT a probability.
          (v0 used raw km; it penalised long gaps so hard that val recall stalled at
          0.68 - see README "Development history".)
learned:  HistGradientBoostingClassifier output in [0, 1] trained on train candidates.
          A ranking score; NOT calibrated confidence (see results/calibration on val).
"""
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42


BASELINE_OFFSET_MIN = 2.0


def baseline_score(f, variant="per_minute"):
    err = (f["fwd_err_km"] + f["back_err_km"]) / 2.0
    if variant == "km":
        return -err
    return -err / (BASELINE_OFFSET_MIN + f["gap_min"])


def to_matrix(candidates, features):
    rows = []
    for _, _, f in candidates:
        rows.append([f[name] if f[name] is not None and not (isinstance(f[name], float) and math.isinf(f[name]))
                     else np.nan for name in features])
    return np.array(rows, dtype=float).reshape(len(candidates), len(features))


def train_model(candidates, true_next, features, seed=SEED):
    X = to_matrix(candidates, features)
    y = np.array([1 if true_next.get(a) == b else 0 for a, b, _ in candidates])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=seed)
    model.fit(X, y)
    return model


def learned_scores(model, candidates, features):
    if not candidates:
        return []
    return model.predict_proba(to_matrix(candidates, features))[:, 1].tolist()

# Failure analysis (test partition, frozen settings)

Source: `results/test_errors.json`, produced by `python -m scripts.evaluate`. I inspected the cases after the settings were frozen and did not change anything in response.

Ground-truth aircraft IDs (`data/_truth/test.jsonl`) were used **only** to categorise errors.

## Error mix at the operating point

| | Baseline | Learned |
|---|---|---|
| Wrong links, other aircraft | 15 | 11 |
| Wrong links, skip (same aircraft, intermediate exists) | 1 | 0 |
| Labelled successor left unresolved | 270 | 33 |
| Tracklets where the two methods disagree | 257 | 257 |

Almost every wrong link involves a **different aircraft**. The skip check removed the skip errors that dominated the baseline during development (see the README).

## Case 1 — Parallel aircraft: both methods accept the neighbour

**`trk-78a9044626 → trk-7f269c7295`** (true successor: `trk-ff2f3b4bc4`)

- **Measurements:**
  - gap 4.9 min;
  - mean projection error 9.5 km;
  - heading change 0.2°;
  - altitude difference 0 ft;
  - no callsign or category available.
- **What happened:** `trk-7f269c7295` belongs to an aircraft flying in parallel about 9 km to the side, at the same speed and altitude. The true continuation `trk-ff2f3b4bc4` comes *after an ~84° turn inside a 16 min gap*, so its straight-line projection error is 67 km forward and 217 km backward.
  - The **baseline** prefers the neighbour by a wide margin (−1.37 vs −7.96 km/min).
  - The **learned** model actually ranks the true successor first (0.436 vs 0.035). However, that same true successor `trk-ff2f3b4bc4` is taken earlier in the greedy pass, so the learned model also falls back to the neighbour, at score 0.035, which is just above its 0.02 threshold.
- **Lesson:** a constant-velocity model cannot separate "same aircraft that turned" from "parallel neighbour that didn't". The very low learned threshold (0.02) let a weak fallback through. A margin or a higher threshold would have withheld it.

## Case 2 — Disagreement: the learned model links through a large turn, wrongly

**`trk-013a7ed57e`** (true successor: `trk-7f269c7295`)

- **Baseline:** unresolved. Its best candidate scores −5.9 km/min, below the −3.0 threshold, so it says it is not confident.
- **Learned:** accepts `trk-78a9044626` with score 0.524. That candidate has a 118° heading change and a 142 km mean projection error. The runner-up scores 0.009.
- **Why:** the training data contains many true links with large turns in the gap, so the model learned that "big heading change + same altitude + consistent category" can still be a continuation. Here the category matched only by chance. With 5 categories, roughly 1 in 3 random pairs match.
- **Lesson:** this is a common learned error pattern: confident on large-manoeuvre pairs when no identifier evidence exists.
  - Measured over all 11 learned wrong links: 10 have `callsign_match = null`; all have a mean projection error of at least 9 km (6 of them above 20 km); 5 scored below 0.06.
  - By contrast, 7 of the 16 baseline wrong links carry an explicit callsign **mismatch** that the physics-only baseline cannot use. This is the clearest evidence that the supporting streams help.

## Case 3 — Tracklet with no successor gets a weak link

**`trk-4936ebb40b`** (label: no successor)

- **Learned:** accepts `trk-aaa2de5f24` at 0.054, with "no other available successor". The pair has gap 7.9 min, forward error 9.6 km, backward error 70.8 km, heading change 45°, and an altitude difference of 4,025 ft.
- **Baseline:** withholds it (−4.06 < −3.0).
- **Lesson:** "only one candidate left" is not evidence. The learned operating point is at the bottom of the validation grid (0.02), where val still showed 0.999 precision. On test, these low-score acceptances are where precision drops to 0.987.

## Case 4 — Missed link: correct candidate beaten by a stronger claim (greedy limitation)

**`trk-00d76408a7`** (true successor: `trk-0686766f10`, gap 42 min)

- **Baseline:** its best candidate `trk-e01cffa4a4` had already been taken by a stronger link (−0.096). The true successor scores −10.3, far below threshold.
- **Learned:** withheld `trk-0686766f10` (0.032) because `trk-e01cffa4a4` fits inside the gap. The skip check says "C is probably not the immediate successor". But `trk-e01cffa4a4` belongs to a different aircraft, so this skip suspicion was a false alarm.
- **Lesson:** the skip check trades a few missed links for fewer skip errors. Validation selected it for both methods.

## Case 5 — Long gap with a large turn: both methods miss (expected)

**`trk-0562d3713e → trk-93f50ed1db`**: a 59.7 min gap, 69° heading change and 948 km forward projection error. **`trk-0587a572bb`** is similar.

- Neither method can recover these from kinematics, and no identifier is present.
- They account for most of the baseline's 270 unresolved cases. We consider leaving them unresolved the correct behaviour.

## Staged scenario observations

- **Learned, stage 1:** accepted `trk-4e94fbf3f4 → trk-2acf0966a9` at 0.401 against a runner-up at **0.401**. That is an effective tie, accepted only because the validation-selected margin is 0.0. It was correct for the stage-1 labels, but by luck.
- **Baseline vs learned, stage 2:** the baseline correctly linked `trk-8a21a02f73 → trk-018b767f69` (projection error 4.3 km, heading change 0.1°). The learned model left it unresolved. The callsign mismatch (`callsign_match = 0`) and a 4,450 ft altitude change pulled its score below threshold.
  - This is a case where trusting a supporting identifier hurts: callsigns are documented as changeable, and the aircraft appears to have climbed during the 25 min gap.

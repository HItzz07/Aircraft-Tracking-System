import copy
import math

import pytest

from src.load import ValidationError, load_observations, load_supporting
from tests.conftest import rec


def load(records):
    return load_observations(records)[0]


# ---------------------------------------------------------------- required fields / IDs
@pytest.mark.parametrize("field", ["observation_id", "tracklet_id", "observed_at", "received_at", "latitude", "longitude", "source"])
def test_missing_required_field(field):
    r = rec(); del r[field]
    with pytest.raises(ValidationError):
        load([r])


@pytest.mark.parametrize("field", ["tracklet_id", "source"])
def test_empty_or_non_string_ids(field):
    for bad in ["", 5, None]:
        r = rec(); r[field] = bad
        with pytest.raises(ValidationError):
            load([r])


def test_empty_observation_id():
    with pytest.raises(ValidationError):
        load([rec(obs_id="")])


def test_ids_not_trimmed_and_case_sensitive():
    out = load([rec(obs_id="o1", tid=" T1 "), rec(obs_id="O1", tid=" T1 ", t="2026-01-10T10:00:10Z")])
    assert list(out) == [" T1 "] and len(out[" T1 "]) == 2


# ---------------------------------------------------------------- coordinates
@pytest.mark.parametrize("lat,lon", [(95, 0), (-90.0001, 0), (0, 180.5), (0, -181), ("10", 0), (True, 0), (math.nan, 0), (0, math.inf)])
def test_bad_coordinates(lat, lon):
    with pytest.raises(ValidationError):
        load([rec(lat=lat, lon=lon)])


def test_coordinate_endpoints_ok():
    assert load([rec(lat=90, lon=-180)])


def test_bad_altitude_type():
    with pytest.raises(ValidationError):
        load([rec(alt="30000")])


def test_attributes_must_be_object():
    r = rec(); r["attributes"] = [1]
    with pytest.raises(ValidationError):
        load([r])


# ---------------------------------------------------------------- timestamps
@pytest.mark.parametrize("ts", ["2026-01-10T10:00:00+05:30", "2026-01-10T10:00:00", "2026-02-30T10:00:00Z",
                                "2026-01-10T10:00:60Z", "2026-01-10T10:00:00.1234Z", "2026-01-10 10:00:00Z", "yesterday"])
def test_bad_timestamps(ts):
    with pytest.raises(ValidationError):
        load([rec(t=ts)])


@pytest.mark.parametrize("ts", ["2026-01-10T10:00:00Z", "2026-01-10T10:00:00+00:00", "2026-01-10T10:00:00.5Z", "2026-01-10T10:00:00.123+00:00"])
def test_good_timestamps(ts):
    assert load([rec(t=ts)])


def test_observed_and_received_are_distinct():
    o = load([rec(t="2026-01-10T10:00:00Z", rx="2026-01-10T11:00:00Z")])["t1"][0]
    assert o["late_receipt_s"] == 3600


# ---------------------------------------------------------------- duplicates and conflicts
def test_identical_duplicate_kept_once_even_with_key_order():
    r = rec()
    reordered = {k: copy.deepcopy(r[k]) for k in reversed(list(r))}
    out, stats = load_observations([r, reordered])
    assert len(out["t1"]) == 1 and stats["n_duplicates_removed"] == 1


def test_int_and_float_equal_values_are_identical():
    a, b = rec(lat=10), rec(lat=10.0)
    assert len(load([a, b])["t1"]) == 1


def test_conflicting_duplicate_rejected_and_named():
    with pytest.raises(ValidationError) as e:
        load([rec(lat=10.0), rec(lat=11.0)])
    assert e.value.observation_id == "o1" and e.value.code == "conflicting_id"
    assert "o1" in str(e.value)


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("altitude_ft"),                         # missing vs null
    lambda r: r["attributes"].update(x=[2, 1]),             # array order matters
    lambda r: r["attributes"].update(flag=True),            # extra field
    lambda r: r.update(attributes={"x": [True, 2]}),        # true vs 1 are different types
])
def test_non_identical_variants_conflict(mutate):
    a, b = rec(), rec()
    mutate(b)
    with pytest.raises(ValidationError):
        load([a, b])


def test_conflict_detected_before_feature_transformations():
    # same parsed instant, different original string -> different content -> conflict
    with pytest.raises(ValidationError):
        load([rec(t="2026-01-10T10:00:00Z"), rec(t="2026-01-10T10:00:00+00:00")])


# ---------------------------------------------------------------- ordering / missing values / preservation
def test_sorted_by_observed_at_not_input_or_receipt_order():
    recs = [rec("o2", t="2026-01-10T10:02:00Z", rx="2026-01-10T10:02:01Z"),
            rec("o1", t="2026-01-10T10:00:00Z", rx="2026-01-10T10:09:00Z"),
            rec("o3", t="2026-01-10T10:01:00Z", rx="2026-01-10T10:01:01Z")]
    assert [o["observation_id"] for o in load(recs)["t1"]] == ["o1", "o3", "o2"]


def test_missing_altitude_stays_none():
    assert load([rec(alt=None)])["t1"][0]["alt_ft"] is None


def test_original_record_preserved():
    r = rec(attrs={"spd": 246, "nested": {"a": [1, None]}})
    original = copy.deepcopy(r)
    o = load([r])["t1"][0]
    assert o["raw"] == original and o["attributes"]["nested"] == {"a": [1, None]}


def test_attributes_optional():
    r = rec(); del r["attributes"]
    assert load([r])["t1"][0]["attributes"] == {}


# ---------------------------------------------------------------- supporting streams
def test_supporting_dedupe_conflict_and_join():
    tr = load([rec()])
    ctx = {"context_id": "c1", "tracklet_id": "t1", "cat": "A3"}
    idr = {"report_id": "r1", "observation_id": "o1", "cs": "GVA12"}
    c, i, stats = load_supporting({"tracklet_context": [ctx, dict(ctx), {"context_id": "c2", "tracklet_id": "zz"}],
                                   "id_reports": [idr]}, tr)
    assert len(c["t1"]) == 1 and len(i["t1"]) == 1 and stats["tracklet_context_unmatched"] == 1
    with pytest.raises(ValidationError):
        load_supporting({"id_reports": [idr, dict(idr, cs="OTHER")]}, tr)

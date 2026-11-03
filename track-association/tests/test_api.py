import json

import joblib
import pytest
from fastapi.testclient import TestClient

import api.main as main
from tests.conftest import rec


@pytest.fixture
def client(tmp_path, model, config, monkeypatch):
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(main, "ART", tmp_path)
    monkeypatch.setattr(main, "RUNS", tmp_path / "runs")
    with TestClient(main.app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_version"] == "test-0"


def test_health_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ART", tmp_path / "missing")
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 503
        r = c.post("/tracklets/associate", json={"observations": [rec()]})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_not_loaded"


def test_restart_reloads_same_artifacts(client, sample):
    body = {"snapshot_id": "x", **sample[0]}
    a = client.post("/tracklets/associate", json=body).json()
    main.load_artifacts()          # simulate a restart
    b = client.post("/tracklets/associate", json=body).json()
    assert a["methods"] == b["methods"] and a["manifest"]["config_hash"] == b["manifest"]["config_hash"]


def test_associate_success_shape(client, sample):
    r = client.post("/tracklets/associate", json={"snapshot_id": "demo", **sample[0]})
    assert r.status_code == 200
    body = r.json()
    assert set(body["methods"]) == {"baseline", "learned"}
    assert {"snapshot_id", "input_hash", "model_version", "config_hash", "decision_config"} <= set(body["manifest"])
    one = next(iter(body["methods"]["learned"].values()))
    assert {"status", "successor", "ranked_candidates", "reason"} <= set(one)
    assert body["tracklets"] and "raw" in next(iter(body["tracklets"].values()))


@pytest.mark.parametrize("payload,status,code", [
    ({"observations": [{"observation_id": "x"}]}, 422, "validation_error"),
    ({"observations": [rec(lat=10.0), rec(lat=11.0)]}, 422, "conflicting_id"),
    ({"observations": []}, 422, "empty_input"),
    ({"observations": "nope"}, 400, "invalid_body"),
    ([1, 2], 400, "invalid_body"),
])
def test_associate_errors(client, payload, status, code):
    r = client.post("/tracklets/associate", json=payload)
    assert r.status_code == status
    e = r.json()["error"]
    assert e["code"] == code and isinstance(e["message"], str)


def test_conflict_error_names_id(client):
    r = client.post("/tracklets/associate", json={"observations": [rec(lat=10.0), rec(lat=11.0)]})
    assert "o1" in r.json()["error"]["message"] and r.json()["error"]["details"]["id"] == "o1"


def test_malformed_json(client):
    r = client.post("/tracklets/associate", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_json"


def test_datasets_listing_and_payload(client):
    names = [d["name"] for d in client.get("/datasets").json()]
    assert "staged/stage_1" in names
    p = client.get("/datasets/staged/stage_1").json()
    assert p["observations"] and "labels" not in p
    assert client.get("/datasets/nope").status_code == 404

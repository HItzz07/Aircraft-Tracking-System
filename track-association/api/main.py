"""FastAPI service. Loads saved artifacts at startup; never trains during a request.

    uvicorn api.main:app --port 8000     # UI at http://localhost:8000
"""
import json
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.evaluate import load_labels, metrics
from src.load import ValidationError, read_jsonl
from src.pipeline import load_folder, run_association

ROOT = Path(__file__).resolve().parents[1]
ART = Path(__import__("os").environ.get("TA_ARTIFACTS", ROOT / "artifacts"))
DATA = ROOT / "data"
RUNS = ROOT / "results" / "api_runs"
STATE = {}


def load_artifacts():
    STATE.clear()
    try:
        STATE["model"] = joblib.load(ART / "model.joblib")
        STATE["config"] = json.loads((ART / "config.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError) as e:
        STATE.clear()
        STATE["load_error"] = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(_app):
    load_artifacts()
    yield


app = FastAPI(title="Track Association", version="1.0.0", lifespan=lifespan)


def err(status, code, message, details=None):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details}})


@app.exception_handler(ValidationError)
def _validation(_req, exc: ValidationError):
    return err(422, exc.code, str(exc), exc.details or None)


@app.exception_handler(RequestValidationError)
def _req_validation(_req, exc):
    return err(400, "invalid_request", "request did not match the expected schema", {"errors": json.loads(json.dumps(exc.errors(), default=str))})


@app.get("/health")
def health():
    ok = "model" in STATE
    cfg = STATE.get("config", {})
    return JSONResponse(status_code=200 if ok else 503, content={
        "status": "ok" if ok else "model_not_loaded",
        "model_version": cfg.get("model_version"),
        "feature_set": cfg.get("feature_set"),
        "artifacts_dir": str(ART),
        "error": STATE.get("load_error"),
    })


@app.post("/tracklets/associate")
async def associate(request: Request):
    """Body: {"snapshot_id": str, "observations": [...], "supporting": {"tracklet_context": [...],
    "id_reports": [...]}, "tracklet_scenarios": {tid: scenario}, "evaluate_against": optional dataset name}."""
    if "model" not in STATE:
        return err(503, "model_not_loaded", "artifacts missing: run `python -m scripts.train` first", {"load_error": STATE.get("load_error")})
    try:
        body = json.loads(await request.body() or b"null")
    except (ValueError, UnicodeDecodeError) as e:
        return err(400, "invalid_json", f"request body is not valid JSON: {e}")
    if not isinstance(body, dict):
        return err(400, "invalid_body", "request body must be a JSON object")
    sid = body.get("snapshot_id", "adhoc")
    if not isinstance(sid, str) or not sid:
        return err(400, "invalid_body", "snapshot_id must be a non-empty string")
    obs = body.get("observations")
    if not isinstance(obs, list):
        return err(400, "invalid_body", "observations must be a list")
    if not obs:
        return err(422, "empty_input", "observations must not be empty")
    resp, cands = run_association(body, STATE["model"], STATE["config"], snapshot_id=sid)
    resp["metrics"], resp["labels"] = None, None
    ev = body.get("evaluate_against")
    if isinstance(ev, str) and ev in datasets_index():
        # labels are attached AFTER decisions, for display only; they never reach the pipeline
        labels = labels_for(ev, set(resp["tracklets"]))
        resp["labels"] = labels
        resp["metrics"] = {m: metrics(resp["methods"][m], labels, cands) for m in ("baseline", "learned")}
    save_run(resp)
    return resp


def save_run(resp):
    RUNS.mkdir(parents=True, exist_ok=True)
    m = resp["manifest"]
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", m["snapshot_id"])[:60]
    out = {"manifest": m, "methods": resp["methods"], "metrics": resp.get("metrics")}
    (RUNS / f"{name}__{m['input_hash']}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


# ------------------------------------------------------------------ supplied datasets for the UI
@lru_cache(maxsize=1)
def datasets_index():
    idx = {}
    for part in ("val", "test"):
        scn = sorted({r["scenario_id"] for r in read_jsonl(DATA / part / "tracklet_index.jsonl")})
        for s in scn:
            idx[f"{part}/{s}"] = {"kind": "partition", "folder": part, "scenario": s, "label": f"{part} · {s}"}
    snaps = json.loads((DATA / "staged" / "snapshots.json").read_text())
    for s in snaps["snapshots"]:
        idx[f"staged/{s['snapshot_id']}"] = {"kind": "staged", "folder": f"staged/{s['folder']}", "scenario": None,
                                             "label": f"staged · {s['snapshot_id']} (received ≤ {s['received_cutoff']})"}
    return idx


@lru_cache(maxsize=8)
def _folder(folder):
    return load_folder(DATA / folder)


def labels_for(name, tids):
    info = datasets_index()[name]
    lab = load_labels(DATA / info["folder"] / "labels.jsonl")
    return {t: lab.get(t) for t in sorted(tids) if t in lab}


@app.get("/datasets")
def list_datasets():
    return [{"name": k, "label": v["label"], "kind": v["kind"]} for k, v in datasets_index().items()]


@app.get("/datasets/{name:path}")
def get_dataset(name: str):
    info = datasets_index().get(name)
    if info is None:
        return err(404, "not_found", f"unknown dataset {name!r}")
    full = _folder(info["folder"])
    if info["scenario"] is None:
        payload = full
    else:
        keep = {t for t, s in full["tracklet_scenarios"].items() if s == info["scenario"]}
        obs = [o for o in full["observations"] if o.get("tracklet_id") in keep]
        oids = {o["observation_id"] for o in obs}
        payload = {"observations": obs,
                   "supporting": {"tracklet_context": [c for c in full["supporting"]["tracklet_context"] if c.get("tracklet_id") in keep],
                                  "id_reports": [r for r in full["supporting"]["id_reports"] if r.get("observation_id") in oids]},
                   "tracklet_scenarios": {t: info["scenario"] for t in keep}}
    return {"snapshot_id": name, "evaluate_against": name, **payload}


@app.get("/results/staged")
def staged_results():
    p = ROOT / "results" / "staged_diff.json"
    if not p.exists():
        return err(404, "not_found", "run `python -m scripts.staged_demo` first")
    return json.loads(p.read_text(encoding="utf-8"))


app.mount("/", StaticFiles(directory=str(ROOT / "api" / "static"), html=True), name="ui")

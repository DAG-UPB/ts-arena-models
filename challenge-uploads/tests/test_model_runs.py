"""The model_runs archive: rows join to the platform schema, and no failure of the archive
can cost a forecast.

The failure cases are the ones that happen on a real host: the directory is gone, the file
is unreadable, the file is corrupt, and another process (an analyst) holds the write lock.
In every case the forecast must still be uploaded and the round settled.
"""
import os
import subprocess
import sys
import textwrap
import time
from datetime import timezone
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_upload_loop import MODELS, OPEN_ROUND, m  # noqa: E402

# round_env patches time.sleep for the uploader, which is the same module object; keep the
# real one for waiting on the lock-holder process.
_real_sleep = time.sleep

METRICS = {"keep_alive": False, "container_start_ms": 4100, "inference_ms": 812,
           "cpu_ms": 1650, "gpu_energy_j": 97.4, "gpu_avg_power_w": 120.0,
           "gpu_idle_power_w": 38.5, "gpu_util_avg": 61.0, "gpu_mem_used_mb_max": 3100,
           "gpu_index": 1, "gpu_name": "NVIDIA RTX A6000", "gpu_foreign_procs": 0,
           "metering_note": None}


@pytest.fixture
def round_env(monkeypatch, tmp_path):
    uploads = []

    def predict(name, histories, horizon, freq, metrics_out=None):
        if metrics_out is not None:
            metrics_out.update(METRICS)
        return [[{"ts": "x", "value": 1.0} for _ in range(3)]]

    monkeypatch.setattr(m, "PARTICIPATION_LOG_FILE", str(tmp_path / "participation.csv"))
    monkeypatch.setattr(m, "get_context_data", lambda rid: [
        {"challenge_series_name": s,
         "data": [{"ts": f"2026-09-22T1{i}:00:00+00:00", "value": float(i)} for i in range(n)]}
        for s, n in (("s", 4), ("t", 6))])
    monkeypatch.setattr(m, "predict_with_model", predict)
    monkeypatch.setattr(m, "upload_forecasts",
                        lambda rid, name, f, deadline=None: uploads.append(name) or {})
    monkeypatch.setattr(m, "USER_ID", "7")
    monkeypatch.setattr(m, "MODEL_IDS", {"container-a": 101, "container-b": 102})
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    return uploads


def _use_db(monkeypatch, path):
    monkeypatch.setattr(m, "MODEL_RUNS", m.ModelRunsWriter(str(path)))


def _rows(path):
    con = duckdb.connect(str(path), read_only=True)
    try:
        cur = con.execute("SELECT * FROM model_runs ORDER BY model_name")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


def test_a_successful_round_writes_joinable_rows(round_env, monkeypatch, tmp_path):
    db = tmp_path / "model_runs.duckdb"
    _use_db(monkeypatch, db)
    assert m.process_challenge(OPEN_ROUND, MODELS) is True

    rows = _rows(db)
    assert [r["model_name"] for r in rows] == ["container-a", "container-b"]
    a = rows[0]
    # join keys: rounds.id, model_info.id, and model_info.name is the CONTAINER name
    assert (a["round_id"], a["model_id"], a["user_id"]) == (11522, 101, 7)
    assert a["config_name"] == "Vendor/A"
    assert a["status"] == "SUCCESS" and a["failed_stage"] is None and a["attempt"] == 0
    # the controller's metrics land verbatim
    assert a["inference_ms"] == 812 and a["gpu_energy_j"] == 97.4 and a["cpu_ms"] == 1650
    # workload shape
    assert (a["n_series"], a["horizon_steps"], a["freq"]) == (2, 3, "h")
    assert (a["context_points"], a["context_points_max"]) == (10, 6)
    # timestamps are tz-aware
    assert a["started_at"].tzinfo is not None
    assert a["finished_at"] >= a["started_at"] and a["total_ms"] >= 0


def test_failures_are_recorded_with_their_stage(round_env, monkeypatch, tmp_path):
    db = tmp_path / "model_runs.duckdb"
    _use_db(monkeypatch, db)

    def outcome(rid, name, f, deadline=None):
        raise m.UploadRejected("unknown series")

    monkeypatch.setattr(m, "upload_forecasts", outcome)
    monkeypatch.setattr(m, "predict_with_model", lambda name, *a, **k: (
        None if name == "container-b" else [[{"ts": "x", "value": 1.0}] * 3]))
    m.process_challenge(OPEN_ROUND, MODELS)
    a, b = _rows(db)
    assert (a["status"], a["failed_stage"]) == ("FAILURE", "upload")
    assert "unknown series" in a["error_message"]
    assert (b["status"], b["failed_stage"]) == ("FAILURE", "predict")


# --- A.2: the archive failing in every realistic way never costs a forecast ----

def _corrupt(path):
    path.write_bytes(os.urandom(4096))


def _unreadable(path):
    duckdb.connect(str(path)).close()
    path.chmod(0)


@pytest.fixture
def locked_db(tmp_path):
    """A second process holding DuckDB's write lock on the file."""
    path = tmp_path / "model_runs.duckdb"
    ready = tmp_path / "ready"
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import duckdb, pathlib, time
        con = duckdb.connect({str(path)!r})
        pathlib.Path({str(ready)!r}).touch()
        time.sleep(60)
    """)])
    for _ in range(300):
        if ready.exists():
            break
        _real_sleep(0.05)
    assert ready.exists(), "lock holder did not start"
    yield path
    holder.kill()
    holder.wait()


@pytest.mark.parametrize("case", ["missing_dir", "unreadable", "corrupt", "locked"])
def test_a_broken_archive_never_costs_a_forecast(case, round_env, monkeypatch, tmp_path,
                                                 request, caplog):
    if case == "missing_dir":
        db = tmp_path / "gone" / "model_runs.duckdb"
    elif case == "locked":
        db = request.getfixturevalue("locked_db")
    else:
        db = tmp_path / "model_runs.duckdb"
        (_corrupt if case == "corrupt" else _unreadable)(db)
    if case == "unreadable" and os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    _use_db(monkeypatch, db)

    submitted = set()
    started = time.monotonic()
    assert m.process_challenge(OPEN_ROUND, MODELS, submitted=submitted) is True
    assert time.monotonic() - started < 5, "a failing archive must not block the round"

    assert round_env == ["container-a", "container-b"], "both forecasts uploaded"
    assert submitted == {"Vendor/A", "Vendor/B"}
    assert "Could not record model run" in caplog.text


def test_an_empty_path_turns_the_archive_off(round_env, monkeypatch):
    monkeypatch.setattr(m, "MODEL_RUNS", m.ModelRunsWriter(""))
    assert m.process_challenge(OPEN_ROUND, MODELS) is True
    assert round_env == ["container-a", "container-b"]


def test_the_archive_recovers_once_the_file_is_fixed(tmp_path):
    db = tmp_path / "model_runs.duckdb"
    writer = m.ModelRunsWriter(str(db))
    row = {"round_id": 1, "model_name": "x", "attempt": 0, "status": "SUCCESS",
           "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:01Z"}
    db.write_bytes(b"not a database")
    assert writer.record(row) is False
    db.unlink()
    assert writer.record(row) is True
    assert writer._failing == 0

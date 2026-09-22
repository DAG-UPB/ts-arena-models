"""Offline tests for the uploader's round loop (models #14).

Synthetic data only — no HTTP, no model containers. They cover the two ways a computed
forecast used to be thrown away:

* **No retry on upload failure.** A ~35 s API outage on 2026-07-22 cost three models their
  round 11522 forecasts. Transient failures must be retried; deterministic ones must not.
* **Settling a round that never submitted.** `process_challenge` returned True regardless
  of whether any upload landed, so the loop retired the round and the forecast was lost.
  A round is settled only once every model has uploaded or the window has closed — and a
  retry must not re-submit the models that already landed.
"""
import importlib.util
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

SRC = Path(__file__).resolve().parents[1] / "src" / "main.py"

# The module creates its log directory at import time and defaults to /app/logs, which
# only exists inside the container. Point it somewhere writable before importing.
os.environ.setdefault("LOG_DIR", tempfile.mkdtemp(prefix="uploader-test-logs-"))


def _load():
    spec = importlib.util.spec_from_file_location("uploader", SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = _load()

BASE = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
FREQ = timedelta(hours=1)
OPEN_ROUND = {
    "id": 11522, "name": "FINGRID", "frequency": "PT1H", "horizon": "PT3H",
    "registration_end": "2099-01-01T00:00:00Z",
}
CLOSED_ROUND = dict(OPEN_ROUND, registration_end="2020-01-01T00:00:00Z")
MODELS = [("container-a", "Vendor/A"), ("container-b", "Vendor/B")]


class Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _accepted(**overrides):
    body = {"success": True, "points_inserted": 6, "probabilistic_points_inserted": 6,
            "errors": [], "warnings": []}
    body.update(overrides)
    return Resp(body)


def _http_error(status):
    resp = requests.Response()
    resp.status_code = status
    return requests.exceptions.HTTPError(f"{status}", response=resp)


FORECASTS = [{"series": "s", "forecasts": [{"ts": "x", "value": 1.0} for _ in range(3)]},
             {"series": "t", "forecasts": [{"ts": "x", "value": 2.0} for _ in range(3)]}]


# --- which upload failures are worth retrying (models #14) -------------------

@pytest.mark.parametrize("exc", [
    requests.exceptions.ConnectionError("refused"),
    requests.exceptions.Timeout("slow"),
    _http_error(502),
    _http_error(500),
])
def test_transient_failures_are_retryable(exc):
    assert m._is_retryable(exc) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    """Including `400 Registration has ended` — retrying only burns the window."""
    assert m._is_retryable(_http_error(status)) is False


def test_upload_retries_a_transient_failure_then_succeeds(monkeypatch):
    calls = []

    def flaky(path, json_data):
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ConnectionError("refused")
        return _accepted()

    monkeypatch.setattr(m, "http_post", flaky)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    assert m.upload_forecasts(1, "mdl", FORECASTS)["points_inserted"] == 6
    assert len(calls) == 3


def test_upload_does_not_retry_a_client_error(monkeypatch):
    calls = []

    def refuse(path, json_data):
        calls.append(1)
        raise _http_error(400)

    monkeypatch.setattr(m, "http_post", refuse)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    with pytest.raises(requests.exceptions.HTTPError):
        m.upload_forecasts(1, "mdl", FORECASTS)
    assert calls == [1]


def test_upload_stops_retrying_once_the_deadline_passed(monkeypatch):
    calls = []

    def down(path, json_data):
        calls.append(1)
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(m, "http_post", down)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    with pytest.raises(requests.exceptions.ConnectionError):
        m.upload_forecasts(1, "mdl", FORECASTS, deadline=past)
    assert calls == [1], "no retry once the window is gone"


# --- the response is verified, not assumed (ts-arena #21, carried across) ----

def test_partially_rejected_201_raises(monkeypatch):
    monkeypatch.setattr(m, "http_post", lambda *a, **k: _accepted(
        points_inserted=3, errors=["Unknown challenge_series_name 't'"]))
    with pytest.raises(m.UploadRejected):
        m.upload_forecasts(1, "mdl", FORECASTS)


def test_short_insert_without_errors_still_raises(monkeypatch):
    monkeypatch.setattr(m, "http_post", lambda *a, **k: _accepted(points_inserted=3))
    with pytest.raises(m.UploadRejected):
        m.upload_forecasts(1, "mdl", FORECASTS)


def test_advisories_alone_do_not_fail_the_upload(monkeypatch):
    monkeypatch.setattr(m, "http_post", lambda *a, **k: _accepted(
        errors=["repaired 2 quantile crossings"], warnings=["repaired 2 quantile crossings"]))
    assert m.upload_forecasts(1, "mdl", FORECASTS)["points_inserted"] == 6


def test_works_against_an_api_portal_without_the_new_fields(monkeypatch):
    monkeypatch.setattr(m, "http_post", lambda *a, **k: Resp(
        {"success": True, "forecasts_inserted": 6}))
    assert m.upload_forecasts(1, "mdl", FORECASTS)["forecasts_inserted"] == 6


# --- a round is settled only by a real submission (models #14) ---------------

@pytest.fixture
def uploader(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "PARTICIPATION_LOG_FILE", str(tmp_path / "participation.csv"))
    monkeypatch.setattr(m, "get_context_data", lambda rid: [
        {"challenge_series_name": "s",
         "data": [{"ts": (BASE + i * FREQ).isoformat(), "value": float(i)} for i in range(4)]}
    ])
    monkeypatch.setattr(m, "predict_with_model",
                        lambda *a, **k: [[{"ts": "x", "value": 1.0} for _ in range(3)]])
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    return m


def test_a_fully_accepted_round_is_settled(uploader, monkeypatch):
    monkeypatch.setattr(m, "upload_forecasts", lambda *a, **k: {"points_inserted": 3})
    submitted = set()
    assert m.process_challenge(OPEN_ROUND, MODELS, submitted=submitted) is True
    assert submitted == {"Vendor/A", "Vendor/B"}


def test_transient_upload_failure_leaves_the_round_open_for_retry(uploader, monkeypatch):
    """Round 11522: the forecast was computed, the upload failed, the round was retired."""
    monkeypatch.setattr(m, "upload_forecasts", lambda *a, **k: (_ for _ in ()).throw(
        requests.exceptions.ConnectionError("refused")))
    assert m.process_challenge(OPEN_ROUND, MODELS) is False


def test_a_refused_upload_does_not_keep_the_round_open(uploader, monkeypatch):
    """Content the platform refused is deterministic. Re-predicting 30 models every poll
    until the window closes would burn the GPU to be refused identically."""
    monkeypatch.setattr(m, "upload_forecasts",
                        lambda *a, **k: (_ for _ in ()).throw(m.UploadRejected("unknown series")))
    refused = set()
    assert m.process_challenge(OPEN_ROUND, MODELS, refused=refused) is True
    assert refused == {"Vendor/A", "Vendor/B"}


def test_a_refused_model_is_not_re_predicted_on_a_later_pass(uploader, monkeypatch):
    """Mixed round: A refused, B transient. Only B comes back."""
    predicted = []
    monkeypatch.setattr(m, "predict_with_model", lambda name, *a, **k: (
        predicted.append(name) or [[{"ts": "x", "value": 1.0} for _ in range(3)]]))

    def outcome(round_id, model_name, forecasts, deadline=None):
        if model_name == "container-a":
            raise m.UploadRejected("unknown series")
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(m, "upload_forecasts", outcome)
    submitted, refused = set(), set()
    assert m.process_challenge(OPEN_ROUND, MODELS,
                               submitted=submitted, refused=refused) is False
    assert refused == {"Vendor/A"} and submitted == set()

    m.process_challenge(OPEN_ROUND, MODELS, submitted=submitted, refused=refused, retry=1)
    assert predicted == ["container-a", "container-b", "container-b"]


def test_retry_does_not_resubmit_the_models_that_already_landed(uploader, monkeypatch):
    calls = []

    def flaky(round_id, model_name, forecasts, deadline=None):
        calls.append(model_name)
        if model_name == "container-b":
            raise requests.exceptions.ConnectionError("transient")
        return {"points_inserted": 3}

    monkeypatch.setattr(m, "upload_forecasts", flaky)
    submitted = set()
    assert m.process_challenge(OPEN_ROUND, MODELS, submitted=submitted) is False
    assert submitted == {"Vendor/A"}

    m.process_challenge(OPEN_ROUND, MODELS, submitted=submitted, retry=1)
    assert calls == ["container-a", "container-b", "container-b"]


def test_no_context_data_is_a_retry_not_a_completion(uploader, monkeypatch):
    monkeypatch.setattr(m, "get_context_data", lambda rid: [])
    assert m.process_challenge(OPEN_ROUND, MODELS) is False


def test_every_retry_path_stops_at_the_deadline(uploader, monkeypatch):
    monkeypatch.setattr(m, "get_context_data", lambda rid: [])
    assert m.process_challenge(CLOSED_ROUND, MODELS) is True

    monkeypatch.setattr(m, "get_context_data", lambda rid: [
        {"challenge_series_name": "s",
         "data": [{"ts": (BASE + i * FREQ).isoformat(), "value": float(i)} for i in range(4)]}
    ])
    monkeypatch.setattr(m, "upload_forecasts", lambda *a, **k: (_ for _ in ()).throw(
        requests.exceptions.ConnectionError("refused")))
    assert m.process_challenge(CLOSED_ROUND, MODELS) is True


def test_a_round_with_no_usable_deadline_counts_as_open():
    assert m.registration_is_open({"id": 1}) is True
    assert m.registration_is_open({"id": 1, "registration_end": "not a date"}) is True
    assert m.registration_deadline({"id": 1}) is None

"""/predict returns the meter's numbers next to the prediction, on both lifecycle paths,
and still stops the container when the prediction fails. Docker is stubbed out."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EVENTS = []


class StubWorker:
    fail = False

    def __init__(self, service_name, base_url, keep_alive=False, **kw):
        self.service_name, self.keep_alive, self.container = service_name, keep_alive, None

    def start(self):
        EVENTS.append("start")

    def stop(self):
        EVENTS.append("stop")

    def predict(self, data=None):
        EVENTS.append("predict")
        if StubWorker.fail:
            raise RuntimeError("model down")
        return {"prediction": [[{"ts": "2026-01-01T00:00:00Z", "value": 1.0}]]}


stub = types.ModuleType("worker")
stub.Worker = StubWorker
stub.client = types.SimpleNamespace(containers=types.SimpleNamespace(
    get=lambda name: (_ for _ in ()).throw(LookupError("no docker here"))))
stub.get_available_models = lambda: []
sys.modules["worker"] = stub

import api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(api.app)
BODY = {"model_name": "naive", "history": [[{"ts": "2026-01-01T00:00:00Z", "value": 1.0}]],
        "horizon": 1, "freq": "h"}


@pytest.fixture(autouse=True)
def reset():
    EVENTS.clear()
    StubWorker.fail = False
    api.kept_alive_models.clear()


def test_cold_path_reports_start_and_inference():
    r = client.post("/predict", json=BODY)
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert m["keep_alive"] is False and m["container_start_ms"] is not None
    assert m["inference_ms"] is not None and m["metering_note"]
    assert EVENTS == ["start", "predict", "stop"]


def test_warm_path_has_no_start():
    api.kept_alive_models.add("naive")
    m = client.post("/predict", json=BODY).json()["metrics"]
    assert m["keep_alive"] is True and m["container_start_ms"] is None
    assert EVENTS == ["predict"]


def test_failed_prediction_still_stops_the_container():
    StubWorker.fail = True
    assert client.post("/predict", json=BODY).status_code == 500
    assert EVENTS == ["start", "predict", "stop"]

"""Offline tests for the inference meter. A fake NVML and a fake container stand in for the
GPU host, so this runs anywhere. The property that matters most: metering can fail in any
way without touching the prediction, and a failing prediction still fails."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import metering  # noqa: E402


class FakeContainer:
    def __init__(self, gpu_ids=("1",), pids=(4242,), cpu_step_ns=250_000_000):
        self._cpu = 0
        self._step = cpu_step_ns
        self.attrs = {"HostConfig": {"DeviceRequests": [
            {"Driver": "nvidia", "DeviceIDs": list(gpu_ids), "Capabilities": [["gpu"]]}]}}
        self._pids = pids

    def stats(self, stream=False, one_shot=True):
        value = self._cpu
        self._cpu += self._step
        return {"cpu_stats": {"cpu_usage": {"total_usage": value}}}

    def top(self):
        return {"Titles": ["UID", "PID", "CMD"],
                "Processes": [["root", str(p), "uvicorn"] for p in self._pids]}


class FakeNVML:
    """Two GPUs, minor numbers 0 and 1. GPU 1 draws 100 W and counts energy in mJ."""

    def __init__(self, foreign_pids=(), energy_supported=True):
        self.foreign = list(foreign_pids)
        self.energy_supported = energy_supported
        self.t0 = time.perf_counter()

    def nvmlInit(self): pass
    def nvmlDeviceGetCount(self): return 2
    def nvmlDeviceGetHandleByIndex(self, i): return i
    def nvmlDeviceGetMinorNumber(self, h): return h
    def nvmlDeviceGetUUID(self, h): return f"GPU-uuid-{h}"
    def nvmlDeviceGetName(self, h): return b"NVIDIA A100"
    def nvmlDeviceGetPowerUsage(self, h): return 100_000
    def nvmlDeviceGetUtilizationRates(self, h): return SimpleNamespace(gpu=80)
    def nvmlDeviceGetMemoryInfo(self, h): return SimpleNamespace(used=2048 * 2**20)

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        if not self.energy_supported:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        return int((time.perf_counter() - self.t0) * 100_000)  # 100 W in mJ/s

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [SimpleNamespace(pid=p) for p in [4242, *self.foreign]]


@pytest.fixture
def nvml(monkeypatch):
    def install(**kw):
        fake = FakeNVML(**kw)
        monkeypatch.setattr(metering, "pynvml", fake)
        monkeypatch.setattr(metering, "_nvml_state", {"ready": None, "error": None})
        return fake
    return install


def _run(container, seconds=0.25):
    meter = metering.InferenceMeter(container, sample_interval=0.02)
    with meter:
        time.sleep(seconds)
    return meter.result


def test_meters_cpu_and_the_containers_gpu(nvml):
    nvml()
    r = _run(FakeContainer())
    assert 230 <= r["inference_ms"] <= 600
    assert r["cpu_ms"] == 250
    assert r["gpu_index"] == 1 and r["gpu_name"] == "NVIDIA A100"
    assert r["gpu_idle_power_w"] == 100.0 and r["gpu_avg_power_w"] == 100.0
    assert r["gpu_energy_j"] == pytest.approx(25, rel=0.4)
    assert r["gpu_util_avg"] == 80.0 and r["gpu_mem_used_mb_max"] == 2048
    assert r["gpu_foreign_procs"] == 0
    assert r["metering_note"] is None


def test_foreign_processes_on_the_gpu_are_counted(nvml):
    nvml(foreign_pids=[9001, 9002])
    assert _run(FakeContainer())["gpu_foreign_procs"] == 2


def test_gpu_is_matched_by_uuid_too(nvml):
    nvml()
    assert _run(FakeContainer(gpu_ids=("GPU-uuid-0",)))["gpu_index"] == 0


def test_energy_falls_back_to_integrated_power(nvml):
    nvml(energy_supported=False)
    r = _run(FakeContainer())
    assert r["gpu_energy_j"] == pytest.approx(25, rel=0.4)
    assert "integrated from power samples" in r["metering_note"]


def test_no_nvml_still_meters_cpu(monkeypatch):
    monkeypatch.setattr(metering, "pynvml", None)
    monkeypatch.setattr(metering, "_nvml_state", {"ready": None, "error": None})
    r = _run(FakeContainer())
    assert r["cpu_ms"] == 250 and "gpu_energy_j" not in r
    assert "not installed" in r["metering_note"]


def test_gpu_not_visible_is_a_note_not_an_error(nvml):
    nvml()
    r = _run(FakeContainer(gpu_ids=("7",)))
    assert "gpu_energy_j" not in r and "not visible" in r["metering_note"]
    assert r["cpu_ms"] == 250


def test_a_broken_container_handle_never_raises(nvml):
    nvml()
    r = _run(None)  # lookup failed upstream
    assert r["inference_ms"] >= 200 and "cpu_ms" not in r
    assert r["metering_note"]


def test_nvml_blowing_up_mid_call_never_raises(nvml):
    fake = nvml()
    meter = metering.InferenceMeter(FakeContainer(), sample_interval=0.02)
    with meter:
        fake.nvmlDeviceGetPowerUsage = lambda h: (_ for _ in ()).throw(RuntimeError("GPU lost"))
        fake.nvmlDeviceGetTotalEnergyConsumption = lambda h: 1 / 0
        time.sleep(0.1)
    assert "GPU lost" in meter.result["metering_note"]
    assert meter.result["cpu_ms"] == 250


def test_the_predictions_own_exception_propagates(nvml):
    nvml()
    meter = metering.InferenceMeter(FakeContainer(), sample_interval=0.02)
    with pytest.raises(ValueError, match="model exploded"):
        with meter:
            raise ValueError("model exploded")
    assert meter.result["inference_ms"] >= 0

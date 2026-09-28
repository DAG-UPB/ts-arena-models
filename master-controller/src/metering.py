"""Resource metering around a single model prediction.

Measures what one warm `/predict` call to a model container costs, as a proxy for the
energy spent on inference:

* wall time of the call,
* CPU time the model container consumed (its cgroup counter, read through the Docker API),
* on the GPU the model container is pinned to: the NVML energy counter delta, plus power,
  utilisation and memory sampled while the call runs.

The GPU is metered per device, not per process, so the meter also counts *foreign* compute
processes on that GPU (anything not in the model container). A non-zero count means the
energy figure includes someone else's work.

Metering is strictly best-effort. Nothing in here may raise into the prediction path: every
failure turns into `None` fields plus a line in `metering_note`, and an exception from the
prediction itself propagates unchanged.
"""
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_S = 0.1

try:
    import pynvml  # provided by nvidia-ml-py
except Exception:  # pragma: no cover - depends on the image
    pynvml = None

_nvml_state: Dict[str, Any] = {"ready": None, "error": None}
_nvml_lock = threading.Lock()


def _nvml_ready() -> bool:
    """Initialise NVML once per process; remember a failure instead of retrying it."""
    with _nvml_lock:
        if _nvml_state["ready"] is None:
            if pynvml is None:
                _nvml_state.update(ready=False, error="nvidia-ml-py not installed")
            else:
                try:
                    pynvml.nvmlInit()
                    _nvml_state["ready"] = True
                except Exception as e:
                    _nvml_state.update(ready=False, error=f"NVML unavailable: {e}")
            if not _nvml_state["ready"]:
                logger.info(f"GPU metering disabled: {_nvml_state['error']}")
        return bool(_nvml_state["ready"])


def container_cpu_ns(container) -> Optional[int]:
    """Cumulative CPU time of a container in nanoseconds (cgroup `total_usage`)."""
    try:
        stats = container.stats(stream=False, one_shot=True)
    except TypeError:  # docker SDK without `one_shot`
        stats = container.stats(stream=False)
    return int(stats["cpu_stats"]["cpu_usage"]["total_usage"])


def container_gpu_ids(container) -> List[str]:
    """The GPU ids the container was created with (`HostConfig.DeviceRequests`)."""
    requests = (container.attrs.get("HostConfig") or {}).get("DeviceRequests") or []
    ids: List[str] = []
    for req in requests:
        ids.extend(str(i) for i in (req.get("DeviceIDs") or []))
    return ids


def container_pids(container) -> Set[int]:
    """Host PIDs of the processes inside the container."""
    top = container.top()
    col = top["Titles"].index("PID")
    return {int(row[col]) for row in top["Processes"]}


def _find_gpu(gpu_ids: List[str]):
    """NVML handle for the GPU the container is pinned to, matched by UUID or by minor
    number (the N in /dev/nvidiaN, which is what Docker's numeric device id selects).
    Matching on the minor number keeps this right whether the controller sees every GPU
    or only its own reservation."""
    if len(gpu_ids) != 1:
        raise LookupError(f"container requests {len(gpu_ids)} GPU ids {gpu_ids}; need exactly one")
    wanted = gpu_ids[0]
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        uuid = pynvml.nvmlDeviceGetUUID(h)
        uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        if wanted.startswith("GPU-"):
            if uuid == wanted:
                return h
        elif str(pynvml.nvmlDeviceGetMinorNumber(h)) == wanted:
            return h
    raise LookupError(f"GPU {wanted} is not visible to the controller")


class InferenceMeter:
    """Context manager that meters the block it wraps. Read `.result` afterwards."""

    def __init__(self, container, sample_interval: float = SAMPLE_INTERVAL_S):
        self.container = container
        self.sample_interval = sample_interval
        self.notes: List[str] = []
        self.result: Dict[str, Any] = {}
        self._gpu = None
        self._own_pids: Set[int] = set()
        self._samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu0: Optional[int] = None
        self._e0: Optional[int] = None
        self._t0 = time.perf_counter()

    # --- helpers -----------------------------------------------------------
    def _note(self, what: str, e: Exception):
        self.notes.append(f"{what}: {type(e).__name__}: {e}")

    def _foreign_procs(self) -> int:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self._gpu)
        return sum(1 for p in procs if p.pid not in self._own_pids)

    def _sample(self) -> Dict[str, float]:
        s: Dict[str, float] = {"t": time.perf_counter()}
        s["power_w"] = pynvml.nvmlDeviceGetPowerUsage(self._gpu) / 1000.0
        s["util"] = float(pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu)
        s["mem_mb"] = pynvml.nvmlDeviceGetMemoryInfo(self._gpu).used / 2**20
        if self._own_pids:
            s["foreign"] = float(self._foreign_procs())
        return s

    def _sampler(self):
        failed = False
        while not self._stop.is_set():
            try:
                self._samples.append(self._sample())
            except Exception as e:
                if not failed:  # one note, not one per tick
                    self._note("gpu sample", e)
                    failed = True
            self._stop.wait(self.sample_interval)

    def _energy_mj(self) -> Optional[int]:
        try:
            return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._gpu))
        except Exception:
            return None

    # --- context manager ---------------------------------------------------
    def __enter__(self):
        try:
            self._cpu0 = container_cpu_ns(self.container)
        except Exception as e:
            self._note("cpu", e)
        try:
            self._gpu_setup()
        except Exception as e:
            self._note("gpu setup", e)
        self._t0 = time.perf_counter()
        return self

    def _gpu_setup(self):
        if not _nvml_ready():
            self.notes.append(_nvml_state["error"])
            return
        try:
            self._gpu = _find_gpu(container_gpu_ids(self.container))
        except Exception as e:
            self._gpu = None
            self._note("gpu", e)
            return
        try:
            self._own_pids = container_pids(self.container)
        except Exception as e:
            self._note("container pids", e)
        try:
            self.result["gpu_index"] = int(pynvml.nvmlDeviceGetMinorNumber(self._gpu))
            name = pynvml.nvmlDeviceGetName(self._gpu)
            self.result["gpu_name"] = name.decode() if isinstance(name, bytes) else name
            self.result["gpu_idle_power_w"] = round(
                pynvml.nvmlDeviceGetPowerUsage(self._gpu) / 1000.0, 2)
        except Exception as e:
            self._note("gpu info", e)
        self._e0 = self._energy_mj()
        self._thread = threading.Thread(target=self._sampler, name="gpu-meter", daemon=True)
        self._thread.start()

    def __exit__(self, exc_type, exc, tb):
        try:
            self._finish()
        except Exception as e:  # never mask or replace the prediction's own outcome
            self._note("finish", e)
        self.result["metering_note"] = "; ".join(self.notes) or None
        return False

    def _finish(self):
        t1 = time.perf_counter()
        elapsed = t1 - self._t0
        self.result["inference_ms"] = int(round(elapsed * 1000))

        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
            e1 = self._energy_mj()
            if self._e0 is not None and e1 is not None and e1 >= self._e0:
                self.result["gpu_energy_j"] = round((e1 - self._e0) / 1000.0, 3)
            samples = self._samples
            if samples:
                powers = [s["power_w"] for s in samples]
                self.result["gpu_avg_power_w"] = round(sum(powers) / len(powers), 2)
                self.result["gpu_util_avg"] = round(
                    sum(s["util"] for s in samples) / len(samples), 1)
                self.result["gpu_mem_used_mb_max"] = int(max(s["mem_mb"] for s in samples))
                foreign = [s["foreign"] for s in samples if "foreign" in s]
                if foreign:
                    self.result["gpu_foreign_procs"] = int(max(foreign))
                if "gpu_energy_j" not in self.result:
                    # No energy counter (pre-Volta, or unsupported): integrate power.
                    self.result["gpu_energy_j"] = round(
                        self.result["gpu_avg_power_w"] * elapsed, 3)
                    self.notes.append("gpu_energy_j integrated from power samples")

        if self._cpu0 is not None:
            try:
                self.result["cpu_ms"] = int(
                    round((container_cpu_ns(self.container) - self._cpu0) / 1e6))
            except Exception as e:
                self._note("cpu", e)

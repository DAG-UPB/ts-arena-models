"""Archive of what each model's prediction cost, one row per (round, model) attempt.

Rows go into a DuckDB file next to the logs. The table is shaped so it joins directly to the
platform database: `round_id` is `challenges.rounds.id`, `model_id` is `models.model_info.id`
and `model_name` is `models.model_info.name`. The columns can later be copied one-to-one
into a TimescaleDB table.

Writing is strictly best-effort. A forecast must never be lost because this file is
missing, locked, full or corrupt, so `record()` catches every exception, logs it, and
returns. The file is opened per write and closed straight after: an analyst can read it
between writes, and a lock held by someone else fails at once instead of blocking the round.
Any other open connection, read-only included, makes writes fail for as long as it is held,
so analysis should run on a copy of the file.
"""
import logging
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

COLUMNS = [
    "run_uuid", "round_id", "model_id", "user_id", "model_name", "config_name",
    "challenge_name", "attempt", "started_at", "finished_at", "total_ms",
    "container_start_ms", "inference_ms", "keep_alive", "cpu_ms", "gpu_energy_j",
    "gpu_avg_power_w", "gpu_idle_power_w", "gpu_util_avg", "gpu_mem_used_mb_max",
    "gpu_index", "gpu_name", "gpu_foreign_procs", "metering_note", "n_series",
    "horizon_steps", "freq", "context_points", "context_points_max", "status",
    "failed_stage", "error_message", "host",
]

DDL = """
CREATE TABLE IF NOT EXISTS model_runs (
    run_uuid            UUID PRIMARY KEY,
    round_id            INTEGER NOT NULL,
    model_id            INTEGER,
    user_id             INTEGER,
    model_name          TEXT NOT NULL,
    config_name         TEXT,
    challenge_name      TEXT,
    attempt             INTEGER NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL,
    finished_at         TIMESTAMPTZ NOT NULL,
    total_ms            INTEGER,
    container_start_ms  INTEGER,
    inference_ms        INTEGER,
    keep_alive          BOOLEAN,
    cpu_ms              INTEGER,
    gpu_energy_j        DOUBLE,
    gpu_avg_power_w     DOUBLE,
    gpu_idle_power_w    DOUBLE,
    gpu_util_avg        DOUBLE,
    gpu_mem_used_mb_max INTEGER,
    gpu_index           INTEGER,
    gpu_name            TEXT,
    gpu_foreign_procs   INTEGER,
    metering_note       TEXT,
    n_series            INTEGER,
    horizon_steps       INTEGER,
    freq                TEXT,
    context_points      INTEGER,
    context_points_max  INTEGER,
    status              TEXT NOT NULL,
    failed_stage        TEXT,
    error_message       TEXT,
    host                TEXT,
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_INSERT = (f"INSERT INTO model_runs ({', '.join(COLUMNS)}) "
           f"VALUES ({', '.join('?' for _ in COLUMNS)})")


class ModelRunsWriter:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._failing = 0  # consecutive failed writes, to log a streak once

    def record(self, row: Dict[str, Any]) -> bool:
        """Insert one row. Returns False on any failure; never raises."""
        if not self.path:
            return False
        try:
            import duckdb

            values = [row.get(c) for c in COLUMNS]
            values[0] = values[0] or str(uuid.uuid4())
            con = duckdb.connect(self.path)
            try:
                con.execute("SET TimeZone = 'UTC'")
                con.execute(DDL)
                con.execute(_INSERT, values)
            finally:
                con.close()
        except Exception as e:
            self._failing += 1
            if self._failing == 1:
                logger.error(f"Could not record model run in {self.path} "
                             f"(forecast unaffected): {type(e).__name__}: {e}")
            else:
                logger.debug(f"model_runs write failed again ({self._failing} in a row): {e}")
            return False

        if self._failing:
            logger.info(f"model_runs writes recovered after {self._failing} failure(s)")
            self._failing = 0
        return True

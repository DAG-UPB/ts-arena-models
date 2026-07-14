# Quantile contract tests & live harness (`model-services/tests/`)

This directory enforces the **structural probabilistic forecast contract** that every
model service in this repo must satisfy. It is deliberately dependency-light and
imports NO model library.

## The contract being checked

Each forecast point a model service returns is a dict:

```json
{"ts": "<str>", "value": <float>, "probabilistic_values": {"q_0.1": float, ..., "q_0.9": float}}
```

- Quantile keys are exactly `q_<level>` for `level` in `0.1, 0.2, ..., 0.9` (deciles).
  Unknown keys (e.g. `q_0.05`, `q_0.95`) are rejected.
- `probabilistic_values: {}` is allowed for point-only models.
- Quantile values must be **finite** (no `NaN`, no `±inf`).
- For a given forecast point, quantile values must be **monotone non-decreasing in level**
  (`q_0.1 ≤ q_0.2 ≤ … ≤ q_0.9`, within `mono_tol=1e-9`).
- When quantiles are present, `value` must equal `q_0.5` (median consistency), within
  `value_tol=1e-6`. If `q_0.5` is absent but other quantiles are present, the median
  check is skipped (it is unusual but not a contract violation).
- A `/predict` response is `{"prediction": [...]}` where the payload is either:
  - **single series**: `List[ForecastItem]`, or
  - **batch**: `List[List[ForecastItem]]`.
  The harness auto-detects which by inspecting `prediction[0]`.
- Each series' length must equal the horizon, and the set of quantile keys must be
  **consistent across all points in a series** (all-or-none per series).

> Sample-based models are intentionally unseeded (issue models-#3): this checker
> asserts STRUCTURAL properties only, **never** exact numeric values.

## Files

| File | Purpose |
|---|---|
| `quantile_contract.py` | Pure-Python checker (stdlib + optional numpy). No model libs. |
| `test_quantile_contract.py` | Offline pytest unit tests, synthetic data only. |
| `run_harness.py` | Live HTTP runner — posts a synthetic history to `/predict` and validates. |
| `__init__.py` | Package marker (empty). |

## Running the offline tests

From the **`ts-arena-models` repo root**:

```bash
uv run --with pytest --with numpy pytest model-services/tests/ -q
```

(Or from inside `model-services/`: `uv run --with pytest --with numpy pytest tests/ -q`.)

Tests must pass offline — no network, no containers.

## Running the live harness

The harness posts a synthetic sine+trend history (48 points) to a `/predict` endpoint
and runs `validate_response` on the returned `prediction`. It exits `0` on PASS,
`1` on contract failure, `2` on request error.

### Direct model service (e.g. chronos, toto)

```bash
uv run model-services/tests/run_harness.py \
    --url http://localhost:8001/predict \
    --horizon 24 --freq h --mode single
```

### Master-controller (requires `--model-name`)

```bash
uv run model-services/tests/run_harness.py \
    --url http://localhost:8080/predict \
    --model-name chronos \
    --horizon 24 --freq h --mode batch
```

### Full CLI options

```
--url URL              /predict URL (required)
--model-name NAME      optional, for master-controller payloads
--horizon N            forecast horizon (default 24)
--freq FREQ            frequency string (default 'h')
--mode {single,batch}  single series or batch (default single)
--timeout SECONDS      HTTP timeout (default 600)
```

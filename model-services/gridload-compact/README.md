# gridload-compact

Compact probabilistic load forecaster for the TS-Arena SMARD net-load challenges (24 h at 15 min, 72 h at 1 h).
Two small MLP quantile networks (one per frequency, a 3-seed ensemble each, well under 1 M parameters in total), numpy feature
pipeline, ONNX Runtime inference on one thread, no torch in the image.  Nine monotone deciles per step; the point value is the median.

## What it is for, and what it is not

- Specialist for the ten SMARD net-load series on the platform's current timestamp convention (UTC stamps that run 1-2 h early, hourly
  series = mean of the four quarter-hours).  A clock-signature gate (`app/model.py::_clock_ok`) checks the daily profile of the context
  against a reference profile (`weights/clock_ref.npz`); only matching, load-like, densely observed contexts reach the nets.
- Everything else (other frequencies, other horizons, other series, short or degenerate contexts) is served by a generic seasonal
  fallback with residual deciles (`app/fallback.py`).  No claim is made for those.
- Holiday calendars (DE federal + 16 state shares, AT) are public deterministic knowledge baked into `weights/hol_table.npz`
  (python `holidays` package, MIT, generated at build time).  An ablation without them is reported in the evaluation.
- Start-up self-check: the service refuses to start when an ONNX net and its feature spec (`weights/net*_info.json`) disagree,
  instead of silently routing every request to the fallback.

## Interface

`POST /predict` as the other services of this repository (single series or batch, `freq` in {`15min`, `h`, ...}); `GET /health`.
`GRIDLOAD_PRECISION=int8` (default) or `fp32` selects the quantized or full-precision graphs; `GRIDLOAD_WEIGHTS` overrides the weights directory.

## Build and test

```bash
docker build -t gridload-compact model-services/gridload-compact
docker run --rm -p 8000:8000 gridload-compact
python model-services/tests/run_harness.py --url http://localhost:8000/predict --horizon 96 --freq 15min --mode batch
# without Docker, from model-services/gridload-compact:  pip install -r requirements.txt pytest httpx && pytest tests -q
```

## Data and licences

- Training data: SMARD (Bundesnetzagentur | SMARD.de, CC BY 4.0) net-load history 2018-01 to the training cutoff, fetched from the public API.
- Evaluation data: TS-Arena-Archive (CC BY 4.0), served contexts, first-published truth, forecasts of the other models.  Results are
  self-obtained evaluations on the Archive, not official platform rankings.
- No third-party model weights were used for training or distillation.  Code and weights: MIT (same licence as this repository).

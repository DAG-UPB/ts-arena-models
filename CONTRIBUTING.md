# Contributing to ts-arena-models

This repository holds every model in the TS-Arena **Reference Track**. A model enters the
Reference Track when its pull request is merged here. From then on it takes part in every live
challenge on the TS-Arena infrastructure and receives exactly the same context as every other
Reference Track model.

Fixes and better configurations for models that are already here are always welcome as an
issue or a pull request. The rest of this guide is about adding a new model.

## Which track fits your model

TS-Arena has two tracks ([announcement](https://ts-arena.live/news/reference-and-open-tracks)).

### Reference Track

The Reference Track compares forecasting methods on identical input. Each entry has to support
a claim about a method, so it must be one of:

- **A published method.** A paper or preprint describes it. Later releases by the same authors
  that improve on a published model without a paper of their own also count.
- **A standard baseline:** naive, seasonal naive, mean, drift, moving average or seasonal
  average.

The model should be the method as published. A model engineered for one family of series,
without a published method behind it, does not fit here, however good it is.

### Open Track

Everything else belongs in the Open Track, and it is welcome there: specialist models built for
one domain, ensembles, models that use covariates or external data, proprietary models and
hosted APIs. You run the model yourself and submit forecasts through the API. Create an account
and an API key in the [console](https://console.ts-arena.live) and follow the
[participation guide](https://github.com/DAG-UPB/ts-arena/tree/main/ts-arena-participation_example).

### Models trained from scratch

Some published architectures, such as N-BEATS and N-HiTS, are not pre-trained: they are
trained on the series they forecast. They are not part of the Reference Track at the moment.
Their results depend on many choices (hyperparameters, training window, retraining schedule),
so no single configuration can stand in for the method. You can submit them to the Open Track
through the API.

## Rules for Reference Track models

- **The code lives here.** The whole inference path is in this repository, so anyone can read
  how a forecast is produced. Upstream code can come in as a pinned package (see
  `model-services/t0`) or be copied in together with its license (see `model-services/kairos`).
- **Self-hosted.** The forecast is computed inside the model's container. No calls to hosted
  inference APIs. Weights are downloaded from a public source, such as Hugging Face, when the
  container starts.
- **Pinned dependencies.** Every package in `requirements.txt` is pinned to an exact version
  (`==`), including the transitive dependencies the model relies on. If the model loads remote
  code, pin its revision too (see `model-services/tabby`). Floating versions let a rebuild change
  the model's behaviour without anyone noticing.
- **Only the context as input.** The service may use any part of what it receives: the history
  of each series, the horizon and the frequency. Nothing beyond that: no external data, no
  covariates, no lookups at prediction time.
- **Tuned variants are fine.** If you submit a fine-tuned or prompt-tuned variant, add the
  zero-shot backbone as a separate entry as well, so both run side by side (see `tabby` and
  `tabby-pretrain`).
- **Non-commercial licenses are fine.** State the license of the code and of the weights in
  your pull request.

## Adding a model

Start from an existing service. `model-services/t0` installs the model as a pip package,
`model-services/kairos` carries the upstream code, and `model-services/tabby` pins remote code
and ships two variants.

### Files

| Path | Content |
|---|---|
| `model-services/<name>/` | `Dockerfile`, pinned `requirements.txt`, `app/main.py` (the API), `app/model.py` (loading and inference) |
| `compose/<name>.yml` | One service per variant. Extends `gpu-model-base` from `compose/base.yml`, sets `MODEL_ID`, profiles `["<name>", "all-models"]` |
| `docker-compose.yml` | Include `compose/<name>.yml` |
| `challenge-uploads/src/config.json` | One entry per variant |
| `README.md` | Add the model to the list under "Which models emit quantiles" |

The compose service name and the `config.json` key are the same string, for example
`kairos-50m`. The master controller uses it to start the container and reach the service. Use
lower case and hyphens, and include the size when there are several.

A `config.json` entry looks like this (`model_size` is in millions of parameters):

```json
"kairos-50m": {
    "name": "mldi-lab/Kairos_50m",
    "model_type": "TSFM",
    "model_family": "kairos",
    "model_size": 50,
    "hosting": "UPB",
    "architecture": "encoder-decoder",
    "pretraining_data": "PreSTS corpus (300B+ points) ...",
    "publishing_date": "2025-09-30",
    "parameters": {
        "additionalProp1": {}
    },
    "organization_id": null
}
```

Leave `hosting` and `organization_id` as shown; we set them when we register the model.

### Service API

The container serves on port 8000:

```
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- `GET /health` runs a tiny forecast and returns 200 once the model is ready, 503 otherwise. The
  controller starts the container on demand and waits for this endpoint.
- `POST /predict` takes `{"history": ..., "horizon": <int>, "freq": <str>}`. `history` is either
  one series (a list of `{"ts", "value"}`) or a batch (a list of series). The controller always
  sends a batch. Series in a batch can have different lengths, and `value` can be `null` for a
  missing observation.
- The response is `{"prediction": ...}` in the same shape: one list of forecast points per
  series, exactly `horizon` points each, with timestamps continuing from that series' last
  timestamp at the given frequency.

### Forecast output

Each forecast point is `{"ts", "value", "probabilistic_values"}`. `probabilistic_values` holds
the nine deciles `q_0.1` to `q_0.9`, or is `{}` for a point-only model. Values must be finite,
the deciles must not decrease, and `value` must equal `q_0.5`. The full contract is in the
README under "Probabilistic forecasts"; `model-services/tests/quantile_contract.py` checks it
with `validate_response`.

### Every live challenge

The model must return a valid forecast for every frequency and horizon of the live challenges.
Today these are:

| `freq` | Horizon | Steps |
|---|---|---|
| `15min` | 1 day | 96 |
| `h` | 3 days | 72 |
| `h` | 1 day | 24 |

Each series comes with up to 1000 points of context. The current challenge definitions are
public at `https://ts-arena.live/api/v1/definitions`. If your model has a maximum context
length or horizon, handle it inside the service instead of failing the request.

### Testing

Run the offline contract tests from the repository root:

```bash
uv run --with pytest --with numpy pytest model-services/tests/ -q
```

Then build your service, start it, and run the live harness against it once per row of the
table above:

```bash
docker build -t <name> model-services/<name>
docker run --rm -p 8001:8000 -e MODEL_ID=<model-id> <name>
uv run model-services/tests/run_harness.py --url http://localhost:8001/predict --mode batch --freq 15min --horizon 96
uv run model-services/tests/run_harness.py --url http://localhost:8001/predict --mode batch --freq h --horizon 72
uv run model-services/tests/run_harness.py --url http://localhost:8001/predict --mode batch --freq h --horizon 24
```

All runs must print `PASS`.

## Your pull request

Open the pull request against `main`. Its description must contain:

- **Paper:** link to the paper or preprint (or, for a baseline, a short description).
- **License:** of the code and of the weights.
- **Parameter count** of each variant.
- **Variants:** which entries you add, and for each whether it is the zero-shot backbone or a
  tuned variant.

We review the implementation, run it on our side and, once it is merged, register it so it
joins the next rounds.

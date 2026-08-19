# TS-Models

Welcome to the **TS-Models** repository! This project is a core component of the **TS-Arena** ecosystem.

## 🏟️ About TS-Arena

TS-Arena is a platform for time series forecasting challenges. Unlike traditional benchmarks on static datasets, TS-Arena challenges participants to predict **live data** into the real future. Evaluations are performed automatically once the ground truth data points become available.

**Main project:** [https://github.com/DAG-UPB/ts-arena](https://github.com/DAG-UPB/ts-arena)

## 🎯 Purpose of this Repository

This repository serves as the initial population for the TS-Arena benchmark. It contains pre-implemented services for various state-of-the-art forecasting models.

While these models are provided by the TS-Arena team to set a baseline, they participate in challenges just like any other external participant.

### 🤝 Call for Feedback
We have implemented these models to the best of our knowledge. However, we highly value input from the original model authors or the community. If you spot any issues with the implementation or have suggestions for better hyperparameters/configurations, please open an issue or a pull request!

## 🚀 Getting Started

### Prerequisites

1.  **Registration:** You must be registered with the TS-Arena platform.
2.  **Environment:** Create a `.env` file in the root directory containing the required environment variables (API keys, credentials, etc.) obtained during registration.

### Running the Models

This setup uses Docker Compose to run the full TS-Arena participation system (including model services and challenge uploads).

#### Recommended approach (best performance):

For optimal memory management, we recommend a two-step process that creates all model containers first (without starting them), then starts only the necessary services:

**For Linux with NVIDIA GPU:**

```bash
# Step 1: Build and create all model containers (stopped, no RAM usage)
docker compose --profile all-models -f docker-compose.yml -f docker-compose.gpu.yml up --build -d --no-start

# Step 2: Start the controller and challenge upload services
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

**For macOS (Apple Silicon or Intel):**

```bash
# Step 1: Build and create all model containers (stopped, no RAM usage)
docker compose --profile all-models -f docker-compose.yml -f docker-compose.macos.yml up --build -d --no-start

# Step 2: Start the controller and challenge upload services
docker compose -f docker-compose.yml -f docker-compose.macos.yml up --build -d
```

**For CPU-only systems:**

```bash
# Step 1: Build and create all model containers (stopped, no RAM usage)
docker compose --profile all-models -f docker-compose.yml up --build -d --no-start

# Step 2: Start the controller and challenge upload services
docker compose -f docker-compose.yml up --build -d
```

*Note: The first step creates all model containers without starting them, minimizing RAM usage. The controller will start model containers on-demand as needed for challenges.*

#### Quick start (single command):

If you prefer a simpler setup, you can start everything at once:

```bash
# Linux with GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

# macOS
docker compose -f docker-compose.yml -f docker-compose.macos.yml up --build -d

# CPU-only
docker compose -f docker-compose.yml up --build -d
```

*Requirements: Linux with GPU requires nvidia-container-toolkit. Set `GPU_ID` in your `.env` file to specify which GPU to use (e.g., `0` or `all`).*

*For local testing without challenge participation, see the `example/` directory for simplified docker-compose configurations.*

## ⚠️ Status

**Current Status: PROTOTYPE**

Please note that this repository is currently in a prototype phase. Breaking changes may occur, and stability is not guaranteed.

## 📂 Repository Structure

- **`compose/`**: Modular docker-compose files for individual models (Chronos, TimesFM, Moirai, Time-MoE, Moment, Sundial, Statistical, NeuralForecast, FlowState, TinyTimeMixer, TireX, VisionTS).
- **`master-controller/`**: Logic for handling challenge tasks and orchestrating predictions.
- **`model-services/`**: Individual container implementations for each forecasting model.
- **`challenge-uploads/`**: Utilities for registering models with the arena.
- **`example/`**: Example docker-compose configurations for local testing without full platform integration.

## 📊 Probabilistic forecasts

Forecasts may carry predictive quantiles in addition to the point `value`. The wire format and invariants below form the **probabilistic forecast contract** every model service in this repo must satisfy. A structural conformance harness lives at `model-services/tests/test_quantile_contract.py`; this README is the human-readable spec it is checked against.

### Wire format

Each forecast point carries a `probabilistic_values` field:

```json
{
  "ts": "2025-01-01T00:00:00Z",
  "value": 42.0,
  "probabilistic_values": {
    "q_0.1": 38.5, "q_0.2": 39.8, "q_0.3": 40.7,
    "q_0.4": 41.4, "q_0.5": 42.0, "q_0.6": 42.6,
    "q_0.7": 43.3, "q_0.8": 44.2, "q_0.9": 45.5
  }
}
```

- The object holds the **nine deciles** `q_0.1 … q_0.9`. Each key is `f"q_{level}"` for level strings `"0.1"` … `"0.9"`.
- For **point-only models**, `probabilistic_values` is the empty object `{}` — this is the intentional, allowed representation of "no quantiles".

### Invariants

For every forecast step where quantiles are present, the following must hold; the conformance harness asserts them structurally:

- **Monotonicity:** quantile values are non-decreasing in level, i.e. `q_0.1 ≤ q_0.2 ≤ … ≤ q_0.9`.
- **Median consistency:** the point `value` equals the median `q_0.5`.

### Which models emit quantiles

After the quantile audit (issue models #13 / models #3), the per-model status is:

- **Point-only (`{}`):** `time-moe`, `moment`, `tinytimemixer`, `tinytimemixer-r1`.
- **Residual-based quantiles** (statistical baselines, derived from point forecast residuals): `naive-forecast`, `seasonal-average`, `simple-moving-average`.
- **Model-native or sample-based quantiles:** `chronos`, `timesfm`, `timesfm2_5`, `moirai`, `moirai2`, `sundial`, `tabpfn-ts`, `tirex`, `flowstate`, `toto`, `visionts`.

Sample-based models are intentionally **unseeded**, so their quantile values vary run-to-run. The conformance harness therefore checks the *structure* and *invariants* of `probabilistic_values`, not exact numeric values (see issue models #3 for the rationale).

### Backend evaluation

The TS-Arena backend scores forecasts as follows:

- A forecast with `probabilistic_values = {}` (point-only) is scored on **point metrics only** (MASE, RMSE).
- A forecast with quantiles present is additionally scored via the **probabilistic metric** — the scaled quantile loss / CRPS.

The probabilistic evaluation itself is defined in the backend under issue **backend #13** ("implement probabilistic evaluation: scaled quantile loss"). Models that emit quantiles here are consequently evaluated on both point and probabilistic metrics on the leaderboard.
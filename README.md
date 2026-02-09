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
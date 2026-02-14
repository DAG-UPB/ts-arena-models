# TS-Models (Prototype)

Welcome to the **TS-Models** repository! This project is a core component of the **TS-Arena** ecosystem.

## 🏟️ About TS-Arena

TS-Arena is a platform for time series forecasting challenges. Unlike traditional benchmarks on static datasets, TS-Arena challenges participants to predict **live data** into the real future. Evaluations are performed automatically once the ground truth data points become available.



## 🎯 Purpose of this Repository

This repository serves as the initial population for the TS-Arena benchmark. It contains pre-implemented services for various state-of-the-art forecasting models.

While these models are provided by the TS-Arena team to set a baseline, they participate in challenges just like any other external participant.

### 🤝 Call for Feedback
We have implemented these models to the best of our knowledge. However, we highly value input from the original model authors or the community. If you spot any issues with the implementation or have suggestions for better hyperparameters/configurations, please open an issue or a pull request!

## 🚀 Getting Started

### Prerequisites

1.  **Docker & Docker Compose:** Ensure Docker and Docker Compose are installed.
2.  **Environment Variables:** Copy `.env.example` to `.env` in the example directory and adjust values as needed:
    - `COMPOSE_PROJECT_NAME`: Project name for Docker Compose (default: `ts-models`)
    - `MASTER_API_PORT`: Port for the master controller API (default: `8456`)
    - `GPU_ID`: (Optional, for GPU setup) GPU device ID to use (default: `0`, or use `all`)

*Note: These example compose files run only the model services locally. For full challenge participation with the TS-Arena platform, use the root-level docker-compose.yml which includes the challenge-uploads service.*

### Running the Models

This setup uses Docker Compose. Choose the appropriate command based on your system:

**For Linux with NVIDIA GPU:**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

*Note: Requires nvidia-container-toolkit. Set `GPU_ID` environment variable to specify which GPU to use (e.g., `0` or `all`).*

**For macOS (Apple Silicon or Intel):**

```bash
docker compose -f docker-compose.yml -f docker-compose.macos.yml up --build -d
```

*Note: Forces linux/amd64 emulation for CUDA-based images when needed.*

**For CPU-only systems:**

```bash
docker compose -f docker-compose.yml up --build -d
```

## ⚠️ Status

**Current Status: PROTOTYPE**

Please note that this repository is currently in a prototype phase. Breaking changes may occur, and stability is not guaranteed.

## 📂 Repository Structure

- **`compose/`**: Modular docker-compose files for individual models (Chronos, TimesFM, Moirai, Time-MoE, Moment, Sundial, Statistical, NeuralForecast, FlowState, TinyTimeMixer, TireX, VisionTS).
- **`master-controller/`**: Logic for handling challenge tasks and orchestrating predictions.
- **`model-services/`**: Individual container implementations for each forecasting model.
- **`challenge-uploads/`**: Utilities for registering models with the arena.
- **`example/`**: Example docker-compose configurations for local testing without full platform integration.

---
*Happy Forecasting!*

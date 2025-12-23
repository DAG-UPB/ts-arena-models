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

1.  **Registration:** You must be registered with the TS-Arena platform.
2.  **Environment:** Create a `.env` file in the root directory containing the required environment variables (API keys, credentials, etc.) obtained during registration.

### Running the Models

This setup uses Docker Compose. The following commands assume you have NVIDIA GPU support enabled.

**1. Build and create the model services (without starting them immediately):**

```bash
docker compose --profile all-models -f docker-compose.yml -f docker-compose.gpu.yml up --build -d --no-start
```

*Note: If you are on macOS or a system without NVIDIA GPUs, you may need to adjust the compose files used (e.g., use `docker-compose.macos.yml` instead of `docker-compose.gpu.yml`).*

**2. Start the Challenge Participation:**

Once the model containers are prepared, start the main controller to begin participating in challenges:

```bash
docker compose -f docker-compose.yml up --build -d
```

## ⚠️ Status

**Current Status: PROTOTYPE**

Please note that this repository is currently in a prototype phase. Breaking changes may occur, and stability is not guaranteed.

## 📂 Repository Structure

- **`compose/`**: Modular docker-compose files for individual models.
- **`master-controller/`**: Logic for handling challenge tasks and orchestrating predictions.
- **`model-services/`**: Individual container implementations for each forecasting model (e.g., Chronos, Moirai, TimesFM, etc.).
- **`challenge-uploads/`**: Utilities for registering models with the arena.

---
*Happy Forecasting!*

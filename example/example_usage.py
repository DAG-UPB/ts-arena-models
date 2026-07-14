import requests
import json
import time
import math
import argparse
import random
from datetime import datetime, timedelta

# Configuration
API_BASE_URL = "http://localhost:8456"

def get_available_models():
    """Fetches the list of available models from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return []

def start_model(model_name):
    """Starts a specific model container."""
    print(f"\n[INFO] Starting model: {model_name}...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/start_model",
            json={"model_name": model_name}
        )
        response.raise_for_status()
        print(f"[SUCCESS] Model {model_name} started.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to start model {model_name}: {e}")
        return False

def stop_model(model_name):
    """Stops the model container."""
    print(f"[INFO] Stopping model: {model_name}...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/stop_model",
            json={"model_name": model_name}
        )
        response.raise_for_status()
        print(f"[SUCCESS] Model {model_name} stopped.")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to stop model {model_name}: {e}")

def generate_dummy_history(length=24):
    """Generates a dummy time series (sine wave)."""
    history = []
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    for i in range(length):
        ts = (start_time - timedelta(hours=length - i)).isoformat() + "Z"
        value = 10 + 5 * math.sin(i * 0.5)
        history.append({"ts": ts, "value": value})
    return history

def generate_complex_history(length=1000, seed=0):
    """Generates a realistic time series with trend, seasonality, and noise.

    Each seed produces a unique series by varying amplitude, frequency,
    trend slope, and noise level.
    """
    rng = random.Random(seed)
    history = []
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    # Randomised parameters per series
    base = rng.uniform(5, 50)
    trend_slope = rng.uniform(-0.005, 0.01)
    amp1 = rng.uniform(2, 10)      # daily-like cycle
    amp2 = rng.uniform(1, 5)       # secondary cycle
    period1 = rng.uniform(20, 30)  # ~daily
    period2 = rng.uniform(150, 250)  # ~weekly
    noise_std = rng.uniform(0.5, 3)

    for i in range(length):
        ts = (start_time - timedelta(hours=length - i)).isoformat() + "Z"
        trend = base + trend_slope * i
        seasonal = amp1 * math.sin(2 * math.pi * i / period1) \
                 + amp2 * math.sin(2 * math.pi * i / period2)
        noise = rng.gauss(0, noise_std)
        value = round(trend + seasonal + noise, 4)
        history.append({"ts": ts, "value": value})
    return history

def run_predictions(model_name, num_predictions=3):
    """Runs multiple predictions against the started model."""
    print(f"[INFO] Running {num_predictions} predictions for {model_name}...")
    
    history_data = generate_dummy_history()
    
    payload = {
        "model_name": model_name,
        "history": history_data,
        "horizon": 12,
        "freq": "h"
    }

    for i in range(num_predictions):
        start_time = time.time()
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=payload)
            response.raise_for_status()
            duration = time.time() - start_time
            result = response.json()
            # Just printing summary to avoid spamming console
            forecast_len = len(result.get("prediction", []))
            print(f"  > Prediction {i+1}: Success ({duration:.2f}s) - Forecast Length: {forecast_len}")
        except requests.exceptions.RequestException as e:
            print(f"  > Prediction {i+1}: Failed inside loop ({e})")

def run_complex_predictions(model_name, num_series=20, context_length=1000, horizon=96):
    """Runs a batch prediction with many long time series.

    Sends `num_series` diverse time series, each with `context_length`
    data points, as a single batch request.
    """
    print(f"\n[COMPLEX] Generating {num_series} time series "
          f"(context={context_length}, horizon={horizon})...")

    gen_start = time.time()
    batch_history = [
        generate_complex_history(length=context_length, seed=i)
        for i in range(num_series)
    ]
    gen_duration = time.time() - gen_start
    print(f"[COMPLEX] Data generated in {gen_duration:.2f}s")

    payload = {
        "model_name": model_name,
        "history": batch_history,
        "horizon": horizon,
        "freq": "h"
    }

    print(f"[COMPLEX] Sending batch prediction request...")
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=600  # generous timeout for large batch
        )
        response.raise_for_status()
        duration = time.time() - start_time
        result = response.json()

        predictions = result.get("prediction", [])
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], list):
                # Batch response: list of lists
                print(f"[COMPLEX] Success ({duration:.2f}s) - "
                      f"{len(predictions)} series forecasted, "
                      f"each with {len(predictions[0])} steps")
            else:
                # Single series response
                print(f"[COMPLEX] Success ({duration:.2f}s) - "
                      f"Forecast length: {len(predictions)}")
        else:
            print(f"[COMPLEX] Success ({duration:.2f}s) - Empty prediction")

    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        print(f"[COMPLEX] Failed ({duration:.2f}s) - {e}")

def main():
    parser = argparse.ArgumentParser(description="Master Controller API Example Consumer")
    parser.add_argument(
        "--complex", action="store_true",
        help="Also run a complex forecasting task: 20 time series with 1000-point context"
    )
    args = parser.parse_args()

    print("=== Master Controller API Example Consumer ===")
    
    # 1. Get Models
    models = get_available_models()
    if not models:
        print("No models found or API unreachable.")
        return

    print(f"Found {len(models)} models: {models}")

    # 2. Iterate over available models
    for model in models:
        # if not "chronos" in model.lower():
        #     continue
        # We can implement a filter here to test only specific models if needed
        if "timesfm-2.5" not in model: continue 

        success = start_model(model)
        
        if success:
            # 3. Make simple predictions
            run_predictions(model, num_predictions=2)

            # 4. Optionally run complex batch predictions
            if args.complex:
                run_complex_predictions(
                    model,
                    num_series=20,
                    context_length=1000,
                    horizon=96
                )
            
            # 5. Stop model
            stop_model(model)
        
        print("-" * 50)

if __name__ == "__main__":
    main()

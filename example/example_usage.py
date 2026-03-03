import requests
import json
import time
import math
from datetime import datetime, timedelta

# Configuration
API_BASE_URL = "http://localhost:7001"

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

def main():
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
        # if "chronos" not in model: continue 

        success = start_model(model)
        
        if success:
            # 3. Make predictions
            run_predictions(model, num_predictions=2)
            
            # 4. Stop model
            stop_model(model)
        
        print("-" * 50)

if __name__ == "__main__":
    main()

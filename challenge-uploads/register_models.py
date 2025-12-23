import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_UPLOAD_KEY")

if not API_BASE_URL:
    print("Error: API_BASE_URL not found in environment variables.")
    exit(1)

if not API_KEY:
    print("Error: API_UPLOAD_KEY not found in environment variables.")
    exit(1)

CONFIG_FILE = "src/config.json"
REGISTER_ENDPOINT = "/api/v1/models/register/admin"

def register_models():
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} not found.")
        return

    try:
        with open(CONFIG_FILE, "r") as f:
            models = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding {CONFIG_FILE}: {e}")
        return

    url = f"{API_BASE_URL}{REGISTER_ENDPOINT}"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    print(f"Registering models to {url}...")

    success_count = 0
    fail_count = 0

    for model_id, model_data in models.items():
        print(f"Registering model: {model_data.get('name', model_id)}...")
        
        try:
            response = requests.post(url, json=model_data, headers=headers)
            
            if response.status_code in [200, 201]:
                print(f"  Success: {model_id}")
                success_count += 1
            else:
                print(f"  Failed: {model_id} - Status: {response.status_code}")
                print(f"  Response: {response.text}")
                fail_count += 1
                
        except requests.RequestException as e:
            print(f"  Error registering {model_id}: {e}")
            fail_count += 1

    print("\nRegistration complete.")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    register_models()
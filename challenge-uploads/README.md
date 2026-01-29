# challenge-uploads: Automatic Challenge Participant

This service automatically monitors available challenges and participates during the registration period. It uses the configured models from the Master Controller to generate and upload predictions.

## How it works

The service performs the following steps:

1. **Challenge Polling**: Regularly polls all available challenges via `GET /api/v1/challenge/`
2. **Registration Check**: Checks for each challenge if the current time is within `registration_start` and `registration_end`
3. **Challenge Details**: Fetches challenge details (`GET /api/v1/challenge/{round_id}`) to determine frequency and horizon
4. **Context Data**: Loads historical data via `GET /api/v1/challenge/{round_id}/context-data` with API key
5. **Prediction**: Sends history data to the Master Controller (`POST http://master-controller:8456/predict`) for each configured model
6. **Formatting**: Formats predictions according to API specification with correct timestamps based on frequency
7. **Upload**: Uploads forecasts via `POST /api/v1/forecasts/upload`

## Configuration

Environment variables in `.env`:

- `API_BASE_URL`: URL of the API Portal (e.g. `http://localhost:8457`)
- `API_UPLOAD_KEY`: API Key for authentication
- `MASTER_CONTROLLER_URL`: URL of the Master Controller (e.g. `http://localhost:8456`)
- `MODEL_MAPPING`: JSON mapping of model IDs to model names
  ```json
  {"1": "timesfm", "2": "sundial", "3": "chronos-bolt", ...}
  ```
- `CHECK_INTERVAL`: Interval in seconds between challenge checks (Default: 60)
- `REQUEST_TIMEOUT`: Timeout for HTTP requests in seconds (Default: 600)
- `LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR)

## Build & Run

### With Docker Compose

```yaml
challenge-uploads:
  build:
    context: ./challenge-uploads
  env_file:
    - .env
  networks:
    - default
  depends_on:
    - api-portal
    - master-controller
```

### Standalone

```bash
cd challenge-uploads
pip install -r requirements.txt
python src/main.py          # Continuous mode (Loop)
python src/main.py once     # One-time run (for testing)
```

## Frequency and Horizon

- **Frequency**: Extracted from `preparation_params["frequency"]` of the challenge (e.g. "15 minutes", "1 hour")
- **Horizon**: Extracted from the `horizon` field of the challenge (e.g. "PT1H" for 1 hour in ISO 8601 format)
- Forecast timestamps are automatically calculated based on frequency

## Processed Challenges

The service remembers already processed challenges (in memory) and skips them on subsequent checks. On service restart, all active registrations are re-processed.

## Logging

- `INFO`: Shows challenge processing and upload status
- `DEBUG`: Detailed HTTP requests and data processing
- `WARNING`: Issues processing individual series
- `ERROR`: Critical errors in API calls or predictions

## Example Output

```
2025-10-23 10:00:00 [INFO] Challenge Upload Service started
2025-10-23 10:00:00 [INFO] API Base URL: http://localhost:8457
2025-10-23 10:00:00 [INFO] Master Controller URL: http://localhost:8456
2025-10-23 10:00:00 [INFO] Check Interval: 60s
2025-10-23 10:00:05 [INFO] Found challenges: 3
2025-10-23 10:00:05 [INFO] Processing challenge 42: Energy Forecast Challenge
2025-10-23 10:00:05 [INFO]   Frequency: 15 minutes -> 0:15:00
2025-10-23 10:00:05 [INFO]   Horizon: PT1H -> 4 steps
2025-10-23 10:00:05 [INFO]   3 series found
2025-10-23 10:00:05 [INFO]   Creating predictions with model timesfm (ID: 1)
2025-10-23 10:00:15 [INFO] ✓ Upload successful for challenge 42, Model 1: 3 series
```

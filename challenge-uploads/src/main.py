import os
import time
import logging
import logging.handlers
import json
import re
import csv
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import requests
from dotenv import load_dotenv
import isodate

# --- Initialization ---
load_dotenv()
time.sleep(2)

# ts-arena #15: `basicConfig(level="info")` raises ValueError -- the obvious
# lowercase spelling of a documented operator knob used to take the service down
# at import. Resolve the level defensively, exactly as ts-arena-backend's
# logging_setup.py does, and never raise.
_LOG_LEVELS = {"CRITICAL": logging.CRITICAL, "FATAL": logging.CRITICAL,
               "ERROR": logging.ERROR, "WARNING": logging.WARNING,
               "WARN": logging.WARNING, "INFO": logging.INFO,
               "DEBUG": logging.DEBUG}
LOG_LEVEL = _LOG_LEVELS.get(os.environ.get("LOG_LEVEL", "").strip().upper(), logging.INFO)
LOG_DIR = os.environ.get("LOG_DIR", "/app/logs")
os.makedirs(LOG_DIR, exist_ok=True)


class _UTCFormatter(logging.Formatter):
    """Formatter whose %(asctime)s is UTC rather than local container time."""

    converter = time.gmtime


# Format matches the rest of the fleet (ts-arena-backend logging_setup.py): the
# old one carried no logger name and rendered local time with no zone marker.
_log_formatter = _UTCFormatter("%(asctime)sZ | %(levelname)s | %(name)s | %(message)s")

# Console handler (existing behaviour)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)

# Rotating file handler — one file per day, keep last 3 days
_file_handler = logging.handlers.TimedRotatingFileHandler(
    filename=os.path.join(LOG_DIR, "challenge-upload.log"),
    when="midnight",
    backupCount=3,
    encoding="utf-8",
    utc=True,
)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(level=LOG_LEVEL, handlers=[_console_handler, _file_handler], force=True)
logging.getLogger("urllib3").setLevel(max(logging.WARNING, LOG_LEVEL))
logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8457")
MASTER_CONTROLLER_URL = os.environ.get("MASTER_CONTROLLER_URL", "http://localhost:8456")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "600"))
API_KEY = os.environ.get("API_UPLOAD_KEY", "default_api_key")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "60"))
USER_ID = os.environ.get("USER_ID")
CONFIG_FILE = os.environ.get("CONFIG_FILE", "config.json")
PARTICIPATION_LOG_FILE = os.environ.get(
    "PARTICIPATION_LOG_FILE", os.path.join(LOG_DIR, "participation_log.csv")
)
LOG_RETENTION_DAYS = int(os.environ.get("LOG_RETENTION_DAYS", "3"))

def log_participation(round_id: str, challenge_name: str, model_container: str,
                      api_model_name: str, status: str, message: str = "",
                      duration_s: Optional[float] = None):
    """Log participation details to CSV file"""
    file_exists = os.path.exists(PARTICIPATION_LOG_FILE)

    try:
        with open(PARTICIPATION_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Challenge ID", "Challenge Name", "Model Container",
                                 "API Model Name", "Status", "Duration (s)", "Message"])

            writer.writerow([
                datetime.now().isoformat(),
                round_id,
                challenge_name,
                model_container,
                api_model_name,
                status,
                f"{duration_s:.2f}" if duration_s is not None else "",
                message
            ])
    except Exception as e:
        logger.error(f"Error writing to participation log: {e}")


def cleanup_participation_log():
    """Remove participation log entries older than LOG_RETENTION_DAYS days."""
    if not os.path.exists(PARTICIPATION_LOG_FILE):
        return

    cutoff = datetime.now() - timedelta(days=LOG_RETENTION_DAYS)
    kept_rows = []
    removed = 0

    try:
        with open(PARTICIPATION_LOG_FILE, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return
            kept_rows.append(header)
            for row in reader:
                if not row:
                    continue
                try:
                    ts = datetime.fromisoformat(row[0])
                    if ts >= cutoff:
                        kept_rows.append(row)
                    else:
                        removed += 1
                except (ValueError, IndexError):
                    kept_rows.append(row)  # keep rows with unparseable timestamps

        with open(PARTICIPATION_LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(kept_rows)

        if removed:
            logger.info(f"Participation log: removed {removed} entries older than {LOG_RETENTION_DAYS} days")
    except Exception as e:
        logger.error(f"Error cleaning up participation log: {e}")

# --- HTTP Helper Functions ---
def http_get(path: str, with_auth: bool = True) -> requests.Response:
    url = f"{API_BASE_URL}{path}"
    headers = {"X-API-Key": API_KEY} if with_auth else {}
    logger.debug(f"GET {url} (auth={with_auth})")
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error beim GET {url}: {e}")
        logger.error(f"  Status Code: {e.response.status_code}")
        logger.error(f"  Response: {e.response.text[:500]}")
        raise


def http_post(path: str, json_data: Dict[str, Any]) -> requests.Response:
    url = f"{API_BASE_URL}{path}"
    logger.debug(f"POST {url}")
    try:
        resp = requests.post(url, json=json_data, timeout=REQUEST_TIMEOUT, headers={"X-API-Key": API_KEY})
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error during POST {url}: {e}")
        logger.error(f"  Status Code: {e.response.status_code}")
        logger.error(f"  Response: {e.response.text[:500]}")
        raise


def master_http_post(path: str, json_data: Dict[str, Any]) -> requests.Response:
    url = f"{MASTER_CONTROLLER_URL}{path}"
    logger.debug(f"MASTER POST {url} payload keys: {json_data.keys()}")
    resp = requests.post(url, json=json_data, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


# --- Model & Config Utils ---
def load_config() -> Dict[str, Any]:
    """Load config file"""
    # Try current dir, script dir and parent dirs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        CONFIG_FILE, 
        os.path.join(script_dir, CONFIG_FILE),
        os.path.join("..", CONFIG_FILE), 
        "/app/config.json"
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    logger.info(f"Loading config from {path}")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config {path}: {e}")
                return {}
    logger.warning(f"No config file found (searched in {paths})")
    return {}


def fetch_registered_models() -> List[Dict[str, Any]]:
    """Fetch registered models from API"""
    if not USER_ID:
        logger.warning("USER_ID not set, cannot fetch models")
        return []
    
    try:
        # Set user_id parameter
        params = {"user_id": USER_ID}
        url = f"{API_BASE_URL}/api/v1/models"
        headers = {"X-API-Key": API_KEY}
        logger.debug(f"GET {url} params={params}")
        
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Error fetching registered models: {e}")
        return []


def resolve_models(config: Dict[str, Any], registered_models: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Match config keys (container names) with registered models.
    Returns: List of (container_name, api_model_name)
    """
    resolved = []
    unmatched = []
    disabled = []

    # Create lookup for registered models by name
    reg_lookup = {m.get("name"): m for m in registered_models}

    for container_name, conf_data in config.items():
        conf_model_name = conf_data.get("name")
        logger.debug(f"Resolving model for container '{container_name}': {conf_model_name}")
        if not conf_model_name:
            continue

        # models #15: a model can be kept in the config but taken out of the running
        # rounds with "enabled": false. Absent flag means enabled, so existing entries
        # are unaffected.
        if not conf_data.get("enabled", True):
            disabled.append(f"{container_name} ({conf_model_name})")
            continue

        if container_name in reg_lookup:
            # Match found
            resolved.append((container_name, conf_model_name))
            logger.debug(f"Model matched: Container '{container_name}' -> API Name '{conf_model_name}'")
        else:
            unmatched.append(f"{container_name} ({conf_model_name})")

    # ts-arena #15: one counted line instead of one warning per unmatched model.
    if unmatched:
        logger.warning(f"{len(unmatched)} model(s) from config not found in API: "
                       + ", ".join(unmatched))

    if disabled:
        logger.info(f"{len(disabled)} model(s) disabled in config: " + ", ".join(disabled))

    return resolved


# --- API Utils ---
def get_all_challenges() -> List[Dict[str, Any]]:
    """Fetch all available challenges (registration phase)"""
    try:
        resp = http_get("/api/v1/challenge/rounds?status=registration", with_auth=True)
        return resp.json() or []
    except Exception as e:
        logger.error(f"Error fetching challenges: {e}")
        return []






def get_context_data(round_id: str) -> List[Dict[str, Any]]:
    """Fetch context data for a challenge round"""
    try:
        resp = http_get(f"/api/v1/challenge/rounds/{round_id}/context-data", with_auth=True)
        return resp.json() or []
    except Exception as e:
        logger.error(f"Error fetching context data for round {round_id}: {e}")
        return []


# --- Frequency Parsing ---
def parse_frequency(frequency_str: str) -> timedelta:
    """Parse frequency string to timedelta (supports ISO 8601 duration)"""
    frequency_str = (frequency_str or "").strip()
    
    # Try ISO 8601 duration first (e.g. 'PT1H', 'PT15M')
    if frequency_str.startswith('P'):
        try:
            return isodate.parse_duration(frequency_str)
        except Exception as e:
            logger.warning(f"Error parsing ISO frequency '{frequency_str}': {e}")
    
    # Legacy / Human-readable formats
    lower_str = frequency_str.lower()
    patterns = [
        (r"(\d+)\s*(?:minute|minutes|min|mins)", lambda m: timedelta(minutes=int(m.group(1)))),
        (r"(\d+)\s*(?:hour|hours|hr|hrs|h)", lambda m: timedelta(hours=int(m.group(1)))),
        (r"(\d+)\s*(?:day|days|d)", lambda m: timedelta(days=int(m.group(1)))),
        (r"(\d+)\s*(?:second|seconds|sec|secs|s)", lambda m: timedelta(seconds=int(m.group(1)))),
    ]
    
    for pattern, converter in patterns:
        match = re.match(pattern, lower_str)
        if match:
            return converter(match)
    
    logger.warning(f"Could not parse frequency '{frequency_str}', using 1 hour as default")
    return timedelta(hours=1)


def parse_horizon(horizon_str: str, frequency) -> int:
    """Parse horizon string (e.g. 'PT1H') to number of steps"""
    try:
        # Use isodate to parse the duration
        duration = isodate.parse_duration(horizon_str)
        
        # Convert duration to seconds
        total_seconds = int(duration.total_seconds())
        
        # Calculate number of steps based on frequency
        step_count = total_seconds // int(frequency.total_seconds())
        
        return step_count
    except Exception as e:
        logger.warning(f"Error parsing horizon string '{horizon_str}': {e}")
        return 1  # Default to 1 step


# --- Context Utils ---
def extract_history_from_context(context_data: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]], List[str], List[datetime]]:
    """
    Extract history data from context data in HistoryItem format
    Returns: (histories, series_names, max_timestamps)
    
    histories is a list of series, where each series is a list of
    HistoryItem dicts: [{"ts": "...", "value": ...}, ...]
    """
    histories = []
    series_names = []
    max_timestamps = []
    
    for serie in context_data:
        name = serie.get('challenge_series_name', f'serie_{len(series_names)}')
        data = serie.get('data', [])
        
        if not data:
            logger.warning(f"Series {name} has no data")
            continue
        
        # Extract as HistoryItem format (ts + value dicts)
        history_items = [{"ts": item['ts'], "value": item['value']} for item in data]
        
        # Find maximum timestamp
        timestamps = [item['ts'] for item in data]
        max_ts_str = max(timestamps)
        max_dt = datetime.fromisoformat(max_ts_str.replace('Z', '+00:00'))
        
        histories.append(history_items)
        series_names.append(name)
        max_timestamps.append(max_dt)
    
    return histories, series_names, max_timestamps


# --- Prediction ---
def predict_with_model(model_name: str, histories: List[List[Dict[str, Any]]], horizon: int, freq: str) -> Optional[List[List[Dict[str, Any]]]]:
    """
    Send predict request to Master Controller
    
    Args:
        model_name: Name of the model
        histories: List of series, each series is a list of HistoryItem dicts
                   [{"ts": "...", "value": ...}, ...]
        horizon: Number of prediction steps
        freq: Frequency string (e.g. "15min", "h", "D")
    
    Returns:
        List of forecast lists or None on error
    """
    if not histories:
        logger.warning(f"No histories for model {model_name} – skipping prediction")
        return None

    payload = {
        "model_name": model_name, 
        "history": histories, 
        "horizon": horizon,
        "freq": freq
    }
    
    try:
        resp = master_http_post("/predict", json_data=payload)
        result = resp.json() or {}
        preds = result.get("prediction")

        if not preds or not isinstance(preds, list):
            logger.warning(f"No valid prediction returned for model {model_name}")
            return None

        return preds
    except Exception as e:
        logger.error(f"Error during prediction with model {model_name}: {e}")
        # Re-raise to be caught by the main loop for logging
        raise


# --- Forecast Formatting ---
def format_forecasts(
    prediction: Union[List[Dict], List[List[Dict]], List[List[float]]], 
    series_names: List[str], 
    max_timestamps: List[datetime],
    frequency_delta: timedelta
) -> List[Dict[str, Any]]:
    """
    Format predictions into upload format
    """
    forecasts_array = []
    
    # Handle different prediction formats
    if isinstance(prediction, list) and len(prediction) > 0:
        first_item = prediction[0]
        
        if isinstance(first_item, dict) and 'ts' in first_item:
            # Single series
            forecasts_array.append({
                "challenge_series_name": series_names[0],
                "forecasts": prediction
             })
        elif isinstance(first_item, list) and len(first_item) > 0 and isinstance(first_item[0], dict) and 'ts' in first_item[0]:
            # Multiple series
            for i, (name, series_forecasts) in enumerate(zip(series_names, prediction)):
                forecasts_array.append({
                    "challenge_series_name": name,
                    "forecasts": series_forecasts
                })
    return forecasts_array


# --- Upload ---
class UploadRejected(Exception):
    """The platform refused part or all of an upload. Deterministic — do not retry."""


def _is_retryable(exc: Exception) -> bool:
    """Is this upload failure worth trying again?

    Connection errors and 5xx are transient — the forecast is already computed, so
    re-POSTing it is cheap (models #14). A 4xx is the server telling us the request is
    wrong, including `400 Registration has ended`; retrying that only wastes the window.
    """
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500
    return False


def upload_forecasts(round_id: int, model_name: str, forecasts: List[Dict[str, Any]],
                     deadline: Optional[datetime] = None) -> Dict[str, Any]:
    """Upload forecasts for a challenge round, retrying transient failures.

    Two things this used to get wrong (models #14):

    * **No retry.** A ~35 s API outage on 2026-07-22 cost three models their round 11522
      forecasts, which had already been computed. Transient failures are now retried with
      backoff, bounded by the round's registration deadline.
    * **No verification.** It logged a tick based on the series it *sent*. A partially
      accepted upload comes back as **HTTP 201** with the rejections in `errors`, so one
      accepted series out of sixteen still read as success. It now parses the response and
      raises `UploadRejected` on anything refused.

    Returns the parsed response body so the caller can log the platform's own numbers.
    """
    payload = {
        "round_id": round_id,
        "model_name": model_name,
        "forecasts": forecasts
    }
    expected_points = sum(len(series["forecasts"]) for series in forecasts)

    backoffs = [2, 5, 10]
    attempt = 0
    while True:
        try:
            resp = http_post("/api/v1/forecasts/upload", json_data=payload)
            break
        except Exception as e:
            out_of_time = deadline is not None and datetime.now(timezone.utc) >= deadline
            if not _is_retryable(e) or attempt >= len(backoffs) or out_of_time:
                reason = ("registration window closed" if out_of_time
                          else "not retryable" if not _is_retryable(e)
                          else "retries exhausted")
                logger.error(
                    f"✗ Upload failed for round {round_id}, model {model_name} ({reason}): {e}"
                )
                raise
            wait = backoffs[attempt]
            attempt += 1
            logger.warning(
                f"  upload attempt {attempt} for round {round_id}, model {model_name} "
                f"failed ({e}); retrying in {wait}s"
            )
            time.sleep(wait)

    try:
        body = resp.json() or {}
    except ValueError:
        logger.warning("Upload response was not JSON; cannot verify what was stored")
        return {}

    # `warnings` and the point/probabilistic split arrive from newer api-portal versions
    # only. Read everything defensively so this keeps working against both.
    errors = body.get("errors") or []
    warnings = body.get("warnings") or []
    rejections = [e for e in errors if e not in warnings]

    inserted = body.get("points_inserted", body.get("forecasts_inserted", 0))
    probabilistic = body.get("probabilistic_points_inserted")

    stored = f"{inserted}/{expected_points} points"
    if probabilistic is not None:
        stored += f", {probabilistic} with quantiles"

    for warning in warnings:
        logger.warning(f"  upload warning: {warning}")

    if rejections or inserted < expected_points:
        for rejection in rejections:
            logger.error(f"  upload rejected: {rejection}")
        raise UploadRejected(
            f"round {round_id}, model {model_name}: stored {stored}"
            + (f"; {len(rejections)} rejection(s): {rejections}" if rejections else "")
        )

    logger.info(f"✓ Upload verified for round {round_id}, model {model_name}: {stored}")
    return body


def registration_deadline(challenge: Dict[str, Any]) -> Optional[datetime]:
    """The round's `registration_end` as an aware datetime, or None if unusable.

    None means "treat the round as open": the uploader's job is to try, and the API is the
    authority on the deadline.
    """
    raw = challenge.get("registration_end")
    if not raw:
        return None
    try:
        deadline = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
    except Exception:
        logger.debug(f"Unparseable registration_end {raw!r}; treating the round as open")
        return None
    return deadline if deadline.tzinfo else deadline.replace(tzinfo=timezone.utc)


def registration_is_open(challenge: Dict[str, Any]) -> bool:
    """Can this round still accept an upload? Used to bound every retry (models #14)."""
    deadline = registration_deadline(challenge)
    return deadline is None or datetime.now(timezone.utc) < deadline


# --- Main ---
def process_challenge(challenge: Dict[str, Any], active_models: List[Tuple[str, str]],
                      submitted: Optional[Set[str]] = None,
                      refused: Optional[Set[str]] = None, retry: int = 0) -> bool:
    """Process a single challenge round.

    Returns True if the round is **settled** — every active model has either uploaded
    successfully or is permanently unprocessable — and False if it should be retried
    later (context data not yet available, or an upload that can still be re-attempted
    inside the registration window).

    `submitted` is the set of model names that already uploaded for this round in an
    earlier pass; it is **mutated in place** as models succeed. Passing it back on a retry
    is what stops a partially successful round from re-uploading the models that already
    landed (models #14) — a duplicate upload returns `points_inserted: 0`, which the
    verification in `upload_forecasts` would otherwise read as a fresh rejection.

    `refused` is the same idea for models the platform **rejected on content** — an unknown
    series name, a wrong point count. Those are deterministic: the identical payload will
    be refused identically, so re-predicting them every poll would burn the GPU for nothing
    (30 models re-running each minute until the window closed). They are dropped from the
    round instead. Only transient failures — a connection error or 5xx that outlived the
    in-call backoff, or a prediction that did not come back — keep the round open.

    `retry` is how many times this round has already come back not-ready. It is a
    logging knob only (ts-arena #15): the round preamble and the "not ready" line
    are worth one INFO the first time and nothing but DEBUG on every 60 s repeat
    after that. The retry itself is intended behaviour, not a bug -- the round is
    deliberately kept out of `processed_challenges` until its context data exists.
    """
    round_id = challenge.get("id")
    challenge_name = challenge.get("name", "Unknown")
    if submitted is None:
        submitted = set()
    if refused is None:
        refused = set()

    # First look at this round gets the full preamble; the repeats go to DEBUG.
    detail = logger.info if retry == 0 else logger.debug
    not_ready = logger.warning if retry == 0 else logger.debug

    if not round_id:
        logger.warning("Skipped challenge without ID")
        return True

    detail(f"Processing challenge round {round_id}: {challenge_name}")

    # Extract frequency and horizon (expected in the rounds response)
    frequency_str = challenge.get("frequency")
    horizon_str = challenge.get("horizon")

    if not frequency_str or not horizon_str:
        logger.warning(f"Challenge round {round_id} missing frequency or horizon")
        return True

    frequency_delta = parse_frequency(frequency_str)
    horizon_steps = parse_horizon(horizon_str, frequency_delta)

    detail(f"  Frequency: {frequency_str} -> {frequency_delta}")
    detail(f"  Horizon: {horizon_str} -> {horizon_steps} steps")

    def wait_for(what: str) -> bool:
        """Retry while the round can still accept an upload; give up once it cannot.

        Without the deadline check a round whose context data never appears is re-polled
        forever (models #14).
        """
        if registration_is_open(challenge):
            not_ready(f"{what} for round {round_id} – will retry in next iteration")
            return False
        logger.error(f"Round {round_id}: registration closed with {what.lower()}")
        return True

    # Fetch context data
    context_data = get_context_data(str(round_id))
    if not context_data:
        return wait_for("No context data")

    # Extract history in HistoryItem format
    histories, series_names, max_timestamps = extract_history_from_context(context_data)
    if not histories:
        return wait_for("No usable history data")

    logger.info(f"  {len(histories)} series found")

    # Convert frequency to model format
    freq_mapping = {
        "1 minute": "1min", "15 minutes": "15min", "30 minutes": "30min",
        "1 hour": "h", "1 day": "D", "1 week": "W", "1 month": "M",
        "PT1M": "1min", "PT15M": "15min", "PT30M": "30min",
        "PT1H": "h", "P1D": "D", "P1W": "W", "P1M": "M"
    }
    model_freq = freq_mapping.get(frequency_str) or freq_mapping.get(frequency_str.lower())
    
    if not model_freq:
        # Generic ISO duration mapping
        if frequency_str.startswith('PT'):
            if 'H' in frequency_str: model_freq = 'h'
            elif 'M' in frequency_str: model_freq = '15min' # default min
        elif frequency_str.startswith('P'):
            if 'D' in frequency_str: model_freq = 'D'
            elif 'W' in frequency_str: model_freq = 'W'
            elif 'M' in frequency_str: model_freq = 'M'
        
        if not model_freq:
            logger.warning(f"Could not map frequency '{frequency_str}' to model format, using 'h'")
            model_freq = 'h'
    
    deadline = registration_deadline(challenge)

    # Process each model that has neither uploaded nor been refused for this round.
    settled_models = submitted | refused
    pending = [(c, a) for c, a in active_models if a not in settled_models]
    if settled_models:
        logger.info(f"  Skipping {len(submitted)} model(s) already uploaded and "
                    f"{len(refused)} the platform refused for this round")

    for container_name, api_model_name in pending:
        logger.info(f"  Creating predictions with container {container_name} for model {api_model_name}")

        t_start: Optional[float] = None
        try:
            # Predict uses container_name — measure inference time
            t_start = time.perf_counter()
            predictions = predict_with_model(container_name, histories, horizon_steps, model_freq)
            duration_s = time.perf_counter() - t_start

            if not predictions:
                logger.warning(f"  No predictions for container {container_name} ({duration_s:.2f}s)")
                log_participation(str(round_id), challenge_name, container_name, api_model_name,
                                  "FAILURE", "Prediction returned None or invalid format",
                                  duration_s=duration_s)
                continue

            logger.info(f"  Prediction done in {duration_s:.2f}s ({container_name})")

            # Format forecasts
            forecasts = format_forecasts(predictions, series_names, max_timestamps, frequency_delta)

            # Upload uses api_model_name
            result = upload_forecasts(int(round_id), container_name, forecasts,
                                      deadline=deadline)
            # Only a verified upload counts as submitted.
            submitted.add(api_model_name)
            # Report what the PLATFORM stored, never what we sent.
            log_participation(
                str(round_id), challenge_name, container_name, api_model_name, "SUCCESS",
                f"Stored {result.get('points_inserted', result.get('forecasts_inserted', '?'))} points "
                f"({result.get('probabilistic_points_inserted', '?')} with quantiles) "
                f"across {len(forecasts)} series",
                duration_s=duration_s)

        except UploadRejected as e:
            # Content the platform refused. Deterministic — do not re-attempt this round.
            duration_s = (time.perf_counter() - t_start) if t_start is not None else None
            refused.add(api_model_name)
            logger.error(f"Upload refused for {container_name}, not retrying this round: {e}")
            log_participation(str(round_id), challenge_name, container_name, api_model_name,
                              "FAILURE", f"refused: {e}", duration_s=duration_s)

        except Exception as e:
            duration_s = (time.perf_counter() - t_start) if t_start is not None else None
            error_details = traceback.format_exc()
            logger.error(f"Error processing model {container_name}: {e}")
            log_participation(str(round_id), challenge_name, container_name, api_model_name,
                              "FAILURE", f"{str(e)}\n{error_details}",
                              duration_s=duration_s)

    outstanding = [a for _, a in active_models if a not in submitted and a not in refused]
    if not outstanding:
        if refused:
            logger.error(
                f"Round {round_id} settled with {len(refused)} model(s) refused "
                f"({', '.join(sorted(refused))})"
            )
        return True

    # Something did not land. Retry it while the round can still accept an upload — this
    # is the half of models #14 that cost round 11522 three models' forecasts: the loop
    # used to mark the round processed regardless, so a computed forecast was thrown away.
    if registration_is_open(challenge):
        logger.warning(
            f"Round {round_id}: {len(outstanding)} model(s) did not upload "
            f"({', '.join(outstanding)}) – will retry while registration is open"
        )
        return False

    logger.error(
        f"Round {round_id}: registration closed with {len(outstanding)} model(s) "
        f"never uploaded ({', '.join(outstanding)})"
    )
    return True


def sleep_until_next_tick(interval: int):
    """
    Sleep until the next wall-clock multiple of `interval`.

    The loop used to end with a plain `time.sleep(interval)`, which measures from the
    moment the work finished rather than from a fixed grid. Processing a round takes
    8-9 minutes, so every processed round pushed the poll phase that much further
    forward and the drift accumulated across days: first contact with a round slid
    from ~3 minutes after its registration opened to ~13 minutes. Against a 15-minute
    registration window that is the difference between the whole roster getting in and
    the tail being refused with "Registration has ended".

    Anchoring to the wall clock keeps the phase fixed no matter how long a round takes.
    """
    delay = interval - (time.time() % interval)
    if delay < 1.0:            # we are already on a tick; wait for the next one
        delay += interval
    time.sleep(delay)


def main_loop():
    """Main loop: Check regularly for new challenges"""
    logger.info("Challenge Upload Service started")
    logger.info(f"API Base URL: {API_BASE_URL}")
    logger.info(f"Master Controller URL: {MASTER_CONTROLLER_URL}")
    logger.info(f"Check Interval: {CHECK_INTERVAL}s")
    logger.info(f"Log directory: {LOG_DIR} (retention: {LOG_RETENTION_DAYS} days)")

    cleanup_participation_log()
    
    # Model initialization
    config = load_config()
    registered_models = fetch_registered_models()
    logger.info(f"Registered models: {len(registered_models)} "
                f"({', '.join(sorted(str(m.get('name')) for m in registered_models)) or 'none'})")
    active_models = resolve_models(config, registered_models)

    if not active_models:
        logger.warning("No active models found. Check config and API.")
    else:
        logger.info(f"Active models: {len(active_models)} – "
                    + ", ".join(f"{c} -> {n}" for c, n in active_models))

    processed_challenges = set()
    # ts-arena #15: rounds that are not ready yet are deliberately re-tried every
    # CHECK_INTERVAL, but they used to re-emit the whole preamble each time. Count
    # the retries so the repeats can drop to DEBUG and the wait is reported once.
    # models #14: the same entry also carries which models already uploaded, so a retry
    # re-attempts only what actually failed.
    pending: Dict[Any, Dict[str, Any]] = {}
    last_challenge_count = None

    while True:
        try:
            # Fetch all challenges
            challenges = get_all_challenges()
            if len(challenges) != last_challenge_count:
                logger.info(f"Found challenges: {len(challenges)}")
                last_challenge_count = len(challenges)
            else:
                logger.debug(f"Found challenges: {len(challenges)} (unchanged)")

            # A round that has dropped out of the registration list can no longer be
            # uploaded to, so stop carrying its retry state.
            open_ids = {c.get("id") for c in challenges}
            for gone in [rid for rid in pending if rid not in open_ids]:
                state = pending.pop(gone)
                logger.error(
                    f"Round {gone} left the registration list after {state['retry']} "
                    f"retries with {len(state['submitted'])} model(s) uploaded"
                )

            for challenge in challenges:
                round_id = challenge.get("id")

                # Check if already processed
                if round_id in processed_challenges:
                    logger.debug(f"Round {round_id} already processed, skipping")
                    continue

                # Process challenge
                state = pending.setdefault(
                    round_id, {"retry": 0, "submitted": set(), "refused": set()})
                retry = state["retry"]
                try:
                    completed = process_challenge(challenge, active_models,
                                                  submitted=state["submitted"],
                                                  refused=state["refused"], retry=retry)
                except Exception as e:
                    # An unexpected error is not evidence the round is done. Leave it
                    # pending so the next poll tries again.
                    logger.error(f"Error processing round {round_id}: {e}")
                    state["retry"] = retry + 1
                    continue

                if completed:
                    processed_challenges.add(round_id)
                    pending.pop(round_id, None)
                    if retry:
                        logger.info(
                            f"Round {round_id} settled after {retry} retries "
                            f"(~{retry * CHECK_INTERVAL}s wait)"
                        )
                else:
                    state["retry"] = retry + 1
                    if retry == 0:
                        logger.info(
                            f"Round {round_id} not ready yet – retrying silently every "
                            f"{CHECK_INTERVAL}s (set LOG_LEVEL=DEBUG to see each attempt)"
                        )

            # Wait for next check
            logger.debug(f"Waiting for the next {CHECK_INTERVAL}s tick...")
            sleep_until_next_tick(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Service stopping...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            sleep_until_next_tick(CHECK_INTERVAL)


def main_once():
    """One-time execution for testing"""
    logger.info("One-time challenge processing")

    cleanup_participation_log()

    # Model initialization
    config = load_config()
    registered_models = fetch_registered_models()
    active_models = resolve_models(config, registered_models)
    
    if not active_models:
        logger.warning("No active models found.")
        return

    challenges = get_all_challenges()
    logger.info(f"Found challenges: {len(challenges)}")
    
    for challenge in challenges:
        # One-shot: there is no next poll, so an unsettled round is reported, not retried.
        if not process_challenge(challenge, active_models):
            logger.warning(
                f"Round {challenge.get('id')} did not fully submit. Re-run, or use the "
                f"service loop, while registration is still open."
            )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        main_once()
    else:
        main_loop()
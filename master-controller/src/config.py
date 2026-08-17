# master-controller/src/config.py

import logging
import os
import json
import sys
import time
from typing import Optional, Dict, Any

def read_secret_file(file_path: str) -> Optional[str]:
    """Reads a secret value from a file (for Docker Secrets)"""
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except (FileNotFoundError, IOError):
        return None

def get_env_or_secret(env_var: str, secret_file_var: str = None) -> Optional[str]:
    """Gets a value from Environment Variable or Docker Secret File"""
    # First try to read from Environment Variable
    value = os.getenv(env_var)
    if value:
        return value
    
    # If not present, try Docker Secret File
    if secret_file_var:
        secret_file_path = os.getenv(secret_file_var)
        if secret_file_path:
            return read_secret_file(secret_file_path)
    
    return None

def load_json_config(env_var: str, secret_file_var: str = None) -> Optional[Dict[str, Any]]:
    """Loads a JSON configuration from Environment Variable or Secret File"""
    config_str = get_env_or_secret(env_var, secret_file_var)
    if config_str:
        try:
            return json.loads(config_str)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON configuration for {env_var}: {e}")
            return None
    return None

class Config:

    # Docker Configuration
    DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "tsfm-arena_default")

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# --- logging (ts-arena #15) -------------------------------------------------
# Config.LOG_LEVEL existed but nothing ever read it: api.py and worker.py both
# hardcoded `basicConfig(level=logging.INFO, ...)` with a format carrying no
# logger name. setup_logging() below is the single place that configures the root
# logger, and it honours LOG_LEVEL. Format and level semantics match
# ts-arena-backend's logging_setup.py so the whole fleet reads alike.

LOG_FORMAT = "%(asctime)sZ | %(levelname)s | %(name)s | %(message)s"

_LOG_LEVELS = {"CRITICAL": logging.CRITICAL, "FATAL": logging.CRITICAL,
               "ERROR": logging.ERROR, "WARNING": logging.WARNING,
               "WARN": logging.WARNING, "INFO": logging.INFO,
               "DEBUG": logging.DEBUG, "NOTSET": logging.NOTSET}

# Libraries that log at INFO and tell an operator nothing they asked for.
NOISY_LIBRARIES = ("docker", "httpcore", "httpx", "urllib3")


class UTCFormatter(logging.Formatter):
    """Formatter whose %(asctime)s is UTC rather than local container time."""

    converter = time.gmtime


def resolve_level(value, default=logging.INFO) -> int:
    """Turn a LOG_LEVEL string into a level int, without ever raising.

    `logging.basicConfig(level="info")` raises ValueError, so the obvious
    lowercase spelling of a documented operator knob must not reach it.
    """
    if isinstance(value, int):
        return value
    if not value:
        return default
    return _LOG_LEVELS.get(str(value).strip().upper(), default)


def setup_logging() -> int:
    """Configure the root logger for the whole process. Returns the resolved level."""
    level = resolve_level(Config.LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(UTCFormatter(LOG_FORMAT))

    root = logging.getLogger()
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Adopt uvicorn's loggers so its lines get the same timestamp, level and name.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(logging.NOTSET)

    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(max(logging.WARNING, level))

    return level


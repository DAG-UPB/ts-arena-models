import os
import time
import logging
from collections import Counter

import httpx
import docker

from config import setup_logging

# ts-arena #15: one place configures the root logger and it honours LOG_LEVEL.
setup_logging()
logger = logging.getLogger(__name__)

TARGET_LABEL = os.environ.get("TARGET_LABEL", "managed.by=controller")

# Prefer from_env so DOCKER_HOST=http://docker-proxy:2375 is used
client = docker.from_env()

class Worker:
    """
    Manages the lifecycle of a Docker container-based prediction worker using Docker SDK.

    This class can be used as a context manager (with a `with` statement),
    to ensure that the Docker container is started before the prediction
    and properly shut down afterwards.
    """
    def __init__(self, service_name: str, base_url: str, port: int = 8000, timeout: float = 300, keep_alive: bool = False):
        """
        Initializes the worker.

        Args:
            service_name (str): The name of the Docker service (e.g. 'chronos-bolt').
            base_url (str): The base URL for the prediction endpoint (e.g. 'http://localhost').
            port (int, optional): The port on which the worker's API runs. Default is 8000.
            timeout (float, optional): Timeout for HTTP requests in seconds. Default is 120.0.
            keep_alive (bool, optional): If True, the container will NOT be stopped on exit. Default is False.
        """
        self.service_name = service_name
        self.base_url = base_url
        self.port = port
        self.predict_url = f"{self.base_url}:{self.port}/predict"
        self.timeout = timeout
        self.container = None
        self.keep_alive = keep_alive

    def __enter__(self):
        """Starts the container and returns the Worker object."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the container when the `with` block is exited."""
        if not self.keep_alive:
            self.stop()

    def start(self):
        """Starts the worker container and waits until it is operational (healthy)."""
        # Find the container by name (assuming container name matches service_name)
        try:
            self.container = client.containers.get(self.service_name)
        except docker.errors.NotFound:
            # The full container list is only worth printing when the lookup failed;
            # it used to be dumped at INFO on every single start().
            logger.error(
                f"Container '{self.service_name}' not found. Containers on this host: "
                f"{[c.name for c in client.containers.list(all=True)]}"
            )
            raise RuntimeError(f"Container '{self.service_name}' not found.")

        if self.container.status != "running":
            self.container.start()

        health_url = f"{self.base_url}:{self.port}/health"
        logger.info(
            f"Starting container '{self.service_name}'; waiting for {health_url} "
            f"(timeout {self.timeout:g}s)"
        )

        # ts-arena #15: this loop used to emit one WARNING per second, each carrying
        # the full response body, for up to `timeout` seconds -- up to ~300 lines to
        # say "still booting", which is the normal case. Aggregate instead (the
        # backend #69 pattern): announce the wait once above, report the outcome
        # once below, and carry the counts so a failure is still diagnosable.
        start_time = time.time()
        probes = 0
        statuses = Counter()
        connect_errors = 0
        last_body = ""

        while time.time() - start_time < self.timeout:
            try:
                probes += 1
                response = httpx.get(health_url, timeout=20)
                if response.status_code == 200:
                    logger.info(
                        f"Container '{self.service_name}' ready after "
                        f"{time.time() - start_time:.1f}s ({probes} health probe(s))"
                    )
                    return

                statuses[response.status_code] += 1
                last_body = response.text[:500]
                time.sleep(1)

            except httpx.RequestError:
                # Normal while starting: the container is not accepting connections yet.
                connect_errors += 1
                time.sleep(1)
            except Exception as e:
                logger.error(
                    f"Unexpected error while waiting for container '{self.service_name}' "
                    f"after {probes} probe(s): {e}"
                )
                self.stop()
                raise

        elapsed = time.time() - start_time
        status_summary = ", ".join(f"{code}x{count}" for code, count in sorted(statuses.items()))
        logger.error(
            f"Container '{self.service_name}' was not reachable after {elapsed:.1f}s: "
            f"{probes} probe(s), {connect_errors} connection error(s), "
            f"statuses [{status_summary or 'none'}]"
            + (f", last body: {last_body}" if last_body else "")
        )
        self.stop()
        raise RuntimeError(f"Timeout waiting for container '{self.service_name}'.")

    def stop(self):
        """Stops the worker container."""
        if self.container is None:
            try:
                self.container = client.containers.get(self.service_name)
            except docker.errors.NotFound:
                logger.info(f"Container '{self.service_name}' not found, nothing to stop.")
                return

        if self.container:
            self.container.reload()
            if self.container.status == "running":
                logger.info(f"Stopping container '{self.service_name}'...")
                self.container.stop(timeout=10)
                logger.info(f"Container '{self.service_name}' stopped.")
            else:
                logger.info(f"Container '{self.service_name}' is not running.")

    def predict(self, data=None):
        """
        Sends a prediction request to the worker.

        Args:
            data (dict, optional): The data for the prediction. If None, a GET request
                                   is sent. Otherwise a POST request with the data as JSON.

        Returns:
            The JSON response from the worker.
        """
        logger.info(f"Requesting prediction from '{self.service_name}'...")
        try:
            if data is None:
                response = httpx.get(self.predict_url, timeout=self.timeout)
            else:
                response = httpx.post(self.predict_url, json=data, timeout=self.timeout)

            response.raise_for_status()
            prediction = response.json()
            logger.info(f"Prediction from '{self.service_name}' received.")
            return prediction
        except httpx.RequestError as e:
            logger.error(f"HTTP request to '{self.service_name}' failed: {e}")
            raise


def list_targets():
    """List all containers with the target label."""
    return client.containers.list(all=True, filters={"label": [TARGET_LABEL]})


def ensure_started():
    """Ensure all target containers are started."""
    for c in list_targets():
        c.reload()
        if c.status != "running":
            logger.info(f"Starting container '{c.name}'...")
            c.start()


def get_available_models() -> list[str]:
    """Returns a list of available model names based on Docker containers."""
    containers = list_targets()
    models = set()
    for container in containers:
        # Use Docker Compose service name as the model name
        service_name = container.labels.get("com.docker.compose.service")
        if service_name:
            models.add(service_name)
    return sorted(list(models))


def ensure_stopped(timeout=10):
    """Ensure all target containers are stopped."""
    for c in list_targets():
        c.reload()
        if c.status == "running":
            logger.info(f"Stopping container '{c.name}'...")
            c.stop(timeout=timeout)


if __name__ == "__main__":
    # Example: Start, wait, stop again
    ensure_started()
    time.sleep(2)
    ensure_stopped()

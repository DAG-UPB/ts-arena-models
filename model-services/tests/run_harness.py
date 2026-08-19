#!/usr/bin/env python3
"""Live harness runner for the TS-Arena quantile contract.

Posts a synthetic history to a /predict endpoint (either a model service directly
or the master-controller), then validates the returned `prediction` with
`quantile_contract.validate_response` and prints a pass/fail summary.

HTTP only — uses the stdlib `urllib.request`. NO model library is imported.

Usage examples:
  # Direct model service
  uv run model-services/tests/run_harness.py \\
      --url http://localhost:8001/predict --horizon 24 --freq h --mode single

  # Master controller (model_name payload required)
  uv run model-services/tests/run_harness.py \\
      --url http://localhost:8080/predict --model-name chronos \\
      --horizon 24 --freq h --mode batch

  # Also works from repo root:
  uv run model-services/tests/run_harness.py --url http://localhost:8001/predict
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta

# Make the sibling quantile_contract importable whether run from repo root or tests/.
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantile_contract  # noqa: E402


def build_history(horizon: int, freq: str, n_points: int = 48) -> list[dict]:
    """Build a synthetic single-series history: sine + linear trend."""
    history = []
    # Pick a sensible step from freq (hours by default).
    step_minutes = {
        "1min": 1,
        "15min": 15,
        "30min": 30,
        "h": 60,
        "D": 60 * 24,
    }.get(freq, 60)

    base = datetime(2026, 1, 1, 0, 0, 0)
    for i in range(n_points):
        ts = (base + timedelta(minutes=step_minutes * i)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
        value = 10.0 + 0.5 * i + 5.0 * math.sin(i / 4.0)
        history.append({"ts": ts, "value": value})
    return history


def build_batch_history(horizon: int, freq: str, n_series: int = 2, n_points: int = 48) -> list[list[dict]]:
    out = []
    for s in range(n_series):
        series = []
        base = datetime(2026, 1, 1, 0, 0, 0)
        step_minutes = {
            "1min": 1, "15min": 15, "30min": 30, "h": 60, "D": 60 * 24,
        }.get(freq, 60)
        for i in range(n_points):
            ts = (base + timedelta(minutes=step_minutes * i)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            )
            value = 10.0 + 0.5 * i + 5.0 * math.sin((i + s * 3) / 4.0) + s
            series.append({"ts": ts, "value": value})
        out.append(series)
    return out


def post_predict(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach {url}: {e}") from None

    if status != 200:
        raise RuntimeError(f"non-200 status {status} from {url}: {body}")

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"response is not JSON: {body[:500]!r}") from e


def run(args: argparse.Namespace) -> int:
    if args.mode == "batch":
        history = build_batch_history(args.horizon, args.freq)
    else:
        history = build_history(args.horizon, args.freq)

    payload = {
        "history": history,
        "horizon": args.horizon,
        "freq": args.freq,
    }
    if args.model_name:
        payload["model_name"] = args.model_name

    print(f"POST {args.url}")
    print(f"  model_name : {args.model_name!r}")
    print(f"  mode       : {args.mode}")
    print(f"  horizon    : {args.horizon}")
    print(f"  freq       : {args.freq!r}")
    print(f"  timeout    : {args.timeout}s")
    print()

    try:
        resp = post_predict(args.url, payload, args.timeout)
    except Exception as e:
        print(f"ERROR (no validation ran): {e}")
        return 2

    prediction = resp.get("prediction")
    if prediction is None:
        print("ERROR: response has no 'prediction' key.")
        print(f"  response keys: {list(resp.keys())}")
        print(f"  body snippet : {json.dumps(resp)[:500]!r}")
        return 2

    result = quantile_contract.validate_response(prediction, horizon=args.horizon)

    # Summary table.
    label = args.url
    if args.model_name:
        label = f"{args.url} (model={args.model_name})"

    print("─" * 78)
    print("QUANTILE CONTRACT RESULT")
    print("─" * 78)
    print(f"{'service':<48} {'passed':<8} {'n_series':<9} {'n_points':<9} {'n_w_q':<6}")
    print("─" * 78)
    print(
        f"{label:<48.48} "
        f"{('PASS' if result['passed'] else 'FAIL'):<8} "
        f"{result['n_series']:<9} "
        f"{result['n_points']:<9} "
        f"{result['n_with_quantiles']:<6}"
    )
    print("─" * 78)
    if result["errors"]:
        print(f"errors ({len(result['errors'])}):")
        for i, e in enumerate(result["errors"], start=1):
            print(f"  {i:>3}. {e}")
    else:
        print("errors: none (all structural checks passed)")
    print()

    return 0 if result["passed"] else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Live quantile-contract harness for TS-Arena model services.")
    p.add_argument("--url", required=True, help="The /predict URL (model service or master-controller).")
    p.add_argument("--model-name", default=None, help="Optional model_name for master-controller payloads.")
    p.add_argument("--horizon", type=int, default=24, help="Forecast horizon (default 24).")
    p.add_argument("--freq", default="h", help="Frequency string (default 'h').")
    p.add_argument("--mode", choices=["single", "batch"], default="single", help="single (one series) or batch (list of series).")
    p.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds (default 600).")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
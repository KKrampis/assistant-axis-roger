#!/usr/bin/env python3
"""
Launch a RunPod GPU pod for the assistant-axis pipeline.

Uses RunPod's REST API (https://rest.runpod.io/v1), verified against its
live OpenAPI spec (https://rest.runpod.io/v1/openapi.json) at the time this
was written. RunPod's offerings (image tags, GPU availability) change over
time -- if pod creation fails, check that spec again before assuming this
script is wrong.

Requires:
    RUNPOD_API_KEY   in the environment (never commit this)
    OPENAI_API_KEY   in the environment (forwarded into the pod's env so
                      pipeline/3_judge.py can run)

Usage:
    export RUNPOD_API_KEY=...
    export OPENAI_API_KEY=...
    python3 runpod/launch_pod.py

    # Then, once the pod is RUNNING, SSH in and run:
    #   bash runpod/bootstrap.sh
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

API_BASE = "https://rest.runpod.io/v1"

# Tried in order; RunPod allocates the first one with available capacity.
GPU_TYPE_IDS = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
]

POD_CONFIG = {
    "name": "assistant-axis-moral-circle",
    "cloudType": "SECURE",
    "gpuTypeIds": GPU_TYPE_IDS,
    "gpuCount": 1,
    # devel (not runtime) image: vLLM's build step can need nvcc/headers.
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "containerDiskInGb": 50,
    # Persistent volume for HF weights (~64GB for Qwen3-32B bf16) + uv cache.
    "volumeInGb": 200,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp"],
}


def api_request(method: str, path: str, api_key: str, body: dict | None = None) -> dict:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        print(f"HTTP {e.code} from {method} {url}:\n{detail}", file=sys.stderr)
        raise


def main():
    runpod_key = os.environ.get("RUNPOD_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not runpod_key:
        print("ERROR: RUNPOD_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set (needed on the pod for step 3).", file=sys.stderr)
        sys.exit(1)

    body = dict(POD_CONFIG)
    body["env"] = {
        "OPENAI_API_KEY": openai_key,
        "HF_HOME": "/workspace/.cache/huggingface",
        "UV_CACHE_DIR": "/workspace/.uv-cache",
    }

    print("Creating pod with config:")
    print(json.dumps({**body, "env": {**body["env"], "OPENAI_API_KEY": "***"}}, indent=2))

    pod = api_request("POST", "/pods", runpod_key, body)
    pod_id = pod.get("id")
    if not pod_id:
        print(f"ERROR: no pod id in response: {pod}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPod created: id={pod_id}")
    print("Save this id -- runpod/stop_pod.py needs it.")
    print(f"\nPolling status (Ctrl+C to stop polling; the pod keeps building)...")

    for _ in range(60):  # ~10 min at 10s intervals
        time.sleep(10)
        status = api_request("GET", f"/pods/{pod_id}", runpod_key)
        desired = status.get("desiredStatus")
        cost = status.get("costPerHr")
        print(f"  desiredStatus={desired} costPerHr={cost}")
        if desired == "RUNNING":
            print("\nPod is running. Full details:")
            print(json.dumps(status, indent=2))
            print(
                f"\nSSH in (check the printed connection details above for host/port), "
                f"then run: bash runpod/bootstrap.sh"
            )
            return

    print("Timed out waiting for RUNNING status -- check the RunPod dashboard.", file=sys.stderr)


if __name__ == "__main__":
    main()

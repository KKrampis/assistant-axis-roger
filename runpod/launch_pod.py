#!/usr/bin/env python3
"""
Launch a RunPod GPU pod for the assistant-axis pipeline.

Uses RunPod's REST API (https://rest.runpod.io/v1), verified against its
live OpenAPI spec (https://rest.runpod.io/v1/openapi.json) at the time this
was written. RunPod's offerings (image tags, GPU availability) change over
time -- if pod creation fails, check that spec again before assuming this
script is wrong.

Requires:
    RUNPOD_API_KEY     in the environment (never commit this)
    OPENAI_API_KEY     in the environment (forwarded into the pod's env so
                        pipeline/3_judge.py can judge with an OpenAI model)

Optional, forwarded into the pod's env if set locally:
    ANTHROPIC_API_KEY  needed on the pod only if you'll judge with a
                        claude-* model (see pipeline/SUBSET_RUNS.md)
    GH_PUSH_TOKEN      a GitHub fine-grained PAT scoped to just this repo
                        (Contents: read/write), so the pod can push results
                        back itself. Deliberately NOT your personal SSH key
                        -- see runpod/README.md "Pushing results from the
                        pod" for why, and how to create one.

Usage:
    export RUNPOD_API_KEY=...
    export OPENAI_API_KEY=...
    python3 runpod/launch_pod.py

    # Then, once the pod is RUNNING, SSH in and run:
    #   bash runpod/bootstrap.sh

    # To re-check connection info for a pod you already launched (e.g. if
    # publicIp/portMappings weren't assigned yet the first time), without
    # creating a new pod:
    python3 runpod/launch_pod.py --status <pod_id>

    # By default, refuses to launch if a pod named with the same prefix is
    # already RUNNING (see POD_NAME_PREFIX) -- this project only ever
    # intends to run one pod at a time. Override with --force if you
    # deliberately want a second one running concurrently.
    python3 runpod/launch_pod.py --force
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

API_BASE = "https://rest.runpod.io/v1"

# Every pod this project creates is named with this prefix -- used both as
# the default pod name and to detect "is one of ours already running" before
# creating a new one. Doesn't touch pods from unrelated projects on the same
# RunPod account.
POD_NAME_PREFIX = "assistant-axis-"
DEFAULT_POD_NAME = f"{POD_NAME_PREFIX}pipeline"

# Tried in order; RunPod allocates the first one with available capacity.
# Secure Cloud pricing at the time this was written (check
# https://www.runpod.io/pricing for current numbers -- these change):
#   A100 80GB PCIe / SXM:  ~$1.59/hr
#   H100 80GB HBM3:        ~$2.89-3.49/hr depending on exact SKU allocated
GPU_TYPE_IDS = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
]

POD_CONFIG = {
    "name": DEFAULT_POD_NAME,
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


def find_running_project_pods(runpod_key: str) -> list[dict]:
    """Return pods whose name starts with POD_NAME_PREFIX and are RUNNING.

    GET /pods returns a bare JSON array (verified against RunPod's OpenAPI
    spec). Doesn't touch or even look closely at pods from unrelated
    projects on the same account -- only used to stop *this* project from
    accidentally running two pods (and paying for two) at once.
    """
    pods = api_request("GET", "/pods", runpod_key)
    return [
        p for p in pods
        if p.get("name", "").startswith(POD_NAME_PREFIX)
        and p.get("desiredStatus") == "RUNNING"
    ]


def print_connection_info(status: dict) -> None:
    public_ip = status.get("publicIp")
    port_mappings = status.get("portMappings") or {}
    ssh_port = port_mappings.get("22")
    if public_ip and ssh_port:
        print(f"\nSSH in with:\n  ssh root@{public_ip} -p {ssh_port}")
    else:
        print(
            "\nWARNING: no publicIp/port-22 mapping yet -- network setup can "
            "lag a few seconds behind desiredStatus=RUNNING. Re-run this "
            "command again in a moment, or check the RunPod dashboard."
        )
    print(f"\nFull pod details:\n{json.dumps(status, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status", metavar="POD_ID", default=None,
        help="Print connection info for an already-launched pod instead of creating a new one.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Launch even if a pod named with the same prefix is already RUNNING.",
    )
    args = parser.parse_args()

    runpod_key = os.environ.get("RUNPOD_API_KEY")
    if not runpod_key:
        print("ERROR: RUNPOD_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    if args.status:
        status = api_request("GET", f"/pods/{args.status}", runpod_key)
        print(f"desiredStatus={status.get('desiredStatus')} costPerHr={status.get('costPerHr')}")
        print_connection_info(status)
        return

    if not args.force:
        existing = find_running_project_pods(runpod_key)
        if existing:
            print("ERROR: a pod from this project is already RUNNING:", file=sys.stderr)
            for p in existing:
                print(f"  id={p.get('id')} name={p.get('name')} costPerHr={p.get('costPerHr')}", file=sys.stderr)
            print(
                "\nThis project only intends to run one pod at a time (each one bills "
                "separately). Stop it first:\n"
                f"  python3 runpod/stop_pod.py {existing[0].get('id')}\n"
                "or pass --force if you deliberately want a second pod running concurrently.",
                file=sys.stderr,
            )
            sys.exit(1)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set (needed on the pod for step 3).", file=sys.stderr)
        sys.exit(1)

    env = {
        "OPENAI_API_KEY": openai_key,
        "HF_HOME": "/workspace/.cache/huggingface",
        "UV_CACHE_DIR": "/workspace/.uv-cache",
    }
    # Optional -- only forwarded if you have them set locally.
    for optional_var in ("ANTHROPIC_API_KEY", "GH_PUSH_TOKEN"):
        val = os.environ.get(optional_var)
        if val:
            env[optional_var] = val

    body = dict(POD_CONFIG)
    body["env"] = env

    print("Creating pod with config:")
    _NOT_SECRET = {"HF_HOME", "UV_CACHE_DIR"}
    redacted_env = {k: (v if k in _NOT_SECRET else "***") for k, v in env.items()}
    print(json.dumps({**body, "env": redacted_env}, indent=2))

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
        print(f"  desiredStatus={desired} costPerHr={cost} publicIp={status.get('publicIp')}")
        if desired == "RUNNING":
            print("\nPod is running.")
            print_connection_info(status)
            print("\nOnce connected, run: bash runpod/bootstrap.sh")
            print(f"(Re-check connection info any time with: python3 runpod/launch_pod.py --status {pod_id})")
            return

    print("Timed out waiting for RUNNING status -- check the RunPod dashboard.", file=sys.stderr)


if __name__ == "__main__":
    main()

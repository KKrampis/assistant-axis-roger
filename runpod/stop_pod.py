#!/usr/bin/env python3
"""
Stop or terminate a RunPod pod, using the same REST API as launch_pod.py.

Usage:
    export RUNPOD_API_KEY=...
    python3 runpod/stop_pod.py <pod_id>              # stop (pause): GPU billing
                                                       # stops, volume persists,
                                                       # resumable later.
    python3 runpod/stop_pod.py <pod_id> --terminate   # delete entirely: also
                                                       # frees the persistent
                                                       # volume. Only do this
                                                       # after pulling results
                                                       # off the pod.
"""

import json
import os
import sys
import urllib.request
import urllib.error

API_BASE = "https://rest.runpod.io/v1"


def api_request(method: str, path: str, api_key: str) -> dict:
    req = urllib.request.Request(f"{API_BASE}{path}", method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        print(f"HTTP {e.code} from {method} {path}:\n{detail}", file=sys.stderr)
        raise


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pod_id = sys.argv[1]
    terminate = "--terminate" in sys.argv[2:]

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    if terminate:
        print(f"Terminating (deleting) pod {pod_id}. This also removes its volume.")
        confirm = input("Type the pod id again to confirm: ")
        if confirm != pod_id:
            print("Confirmation did not match, aborting.")
            sys.exit(1)
        api_request("DELETE", f"/pods/{pod_id}", api_key)
        print("Terminated.")
    else:
        print(f"Stopping pod {pod_id} (volume persists, GPU billing stops).")
        result = api_request("POST", f"/pods/{pod_id}/stop", api_key)
        print(json.dumps(result, indent=2) if result else "Stopped.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import sys


LAYER_ORDER = [
    "service_process",
    "listener",
    "http",
    "authenticated_api",
    "admin_identity",
    "mutating_smoke",
    "persistence",
]


def load_checks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise SystemExit("checks json must be a list")
    return payload


def summarize(checks: list[dict]) -> dict:
    passed = []
    failed = None
    normalized = {item.get("layer"): item for item in checks if item.get("layer")}
    for layer in LAYER_ORDER:
        item = normalized.get(layer)
        if item is None:
            continue
        ok = bool(item.get("ok"))
        if ok and failed is None:
            passed.append(layer)
            continue
        if not ok and failed is None:
            failed = {
                "layer": layer,
                "cause": item.get("cause", "unknown"),
                "detail": item.get("detail", ""),
            }
            break
    if failed is None:
        return {"state": "executed", "passed_layers": passed, "failed_layer": None}
    return {
        "state": "needs_repair" if passed else "blocked",
        "passed_layers": passed,
        "failed_layer": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize layered service acceptance checks.")
    parser.add_argument("--checks-json", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(load_checks(args.checks_json)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

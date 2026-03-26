#!/usr/bin/env python3
import argparse
import json
import secrets
from pathlib import Path


def parse_kv(items: list[str]) -> dict[str, str]:
    values = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"invalid KEY=VALUE item: {item}")
        values[key] = value
    return values


def write_env(path: Path, values: dict[str, str]) -> None:
    existing = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                existing[key] = value
    existing.update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(existing.items())),
        encoding="utf-8",
    )
    path.chmod(0o600)


def describe(path: Path) -> dict:
    keys = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                keys.append(line.split("=", 1)[0])
    return {"path": str(path), "exists": path.exists(), "keys": sorted(keys)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Open helper for local secret material references.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate-password")
    gen.add_argument("--length", type=int, default=32)
    upsert = sub.add_parser("upsert-env")
    upsert.add_argument("--path", required=True)
    upsert.add_argument("--set", action="append", default=[])
    desc = sub.add_parser("describe")
    desc.add_argument("--path", required=True)
    args = parser.parse_args()
    if args.command == "generate-password":
        print(json.dumps({"password": secrets.token_hex(max(1, args.length // 2))}, indent=2))
        return 0
    if args.command == "upsert-env":
        path = Path(args.path)
        values = parse_kv(args.set)
        write_env(path, values)
        print(json.dumps(describe(path), indent=2))
        return 0
    print(json.dumps(describe(Path(args.path)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

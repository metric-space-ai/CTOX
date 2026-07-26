#!/usr/bin/env python3
"""Normalize an independently audited source catalog for CTOX import.

The input remains authoritative for titles, URLs, relevance, and audit status.
This adapter verifies every local snapshot byte-for-byte before translating
external verification labels into the strict Business OS Evidence-v2 contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


class NormalizeError(ValueError):
    """The external catalog cannot be admitted fail-closed."""


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise NormalizeError("invalid_snapshot_sha256")
    return text


def normalize_row(
    row: dict[str, str],
    *,
    source_root: Path,
    target_snapshot_root: str,
    research_run_id: str,
    research_command_id: str,
) -> dict[str, Any]:
    source_id = str(row.get("source_id") or "").strip()
    status = str(row.get("verification_status") or "").strip().lower()
    rejection = str(row.get("rejection_reason") or "").strip()
    if not source_id:
        raise NormalizeError("missing_source_id")
    if not status.startswith("verified_"):
        raise NormalizeError(f"{source_id}:verification_not_audited")
    if not truthy(row.get("evidence_eligible")) or rejection:
        raise NormalizeError(f"{source_id}:not_evidence_eligible")
    try:
        http_status = int(row.get("http_status") or "")
        relevance_original = Decimal(str(row.get("evidence_relevance_score") or ""))
        relevance = int(
            relevance_original.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
    except (InvalidOperation, ValueError) as exc:
        raise NormalizeError(f"{source_id}:invalid_numeric_audit_field") from exc
    if http_status < 200 or http_status >= 300 or http_status == 204:
        raise NormalizeError(f"{source_id}:http_status_not_usable")
    if relevance < 8:
        raise NormalizeError(f"{source_id}:relevance_below_threshold")

    relative_snapshot = Path(str(row.get("snapshot_path") or ""))
    if relative_snapshot.is_absolute() or ".." in relative_snapshot.parts:
        raise NormalizeError(f"{source_id}:unsafe_snapshot_path")
    snapshot_path = source_root / relative_snapshot
    if not snapshot_path.is_file():
        raise NormalizeError(f"{source_id}:snapshot_missing:{relative_snapshot}")
    expected_hash = normalized_hash(row.get("snapshot_sha256"))
    if sha256_file(snapshot_path) != expected_hash:
        raise NormalizeError(f"{source_id}:snapshot_hash_mismatch")

    canonical_url = str(row.get("canonical_url") or "").strip()
    if not canonical_url.startswith(("https://", "http://")):
        raise NormalizeError(f"{source_id}:canonical_url_invalid")
    archive_verified = "archive_tested" in status
    evidence_id = f"EVID-SOURCE-{source_id.removeprefix('SRC-')}"
    target_path = f"{target_snapshot_root.rstrip('/')}/{relative_snapshot.as_posix()}"

    normalized = dict(row)
    normalized.update(
        {
            "research_run_id": research_run_id,
            "research_command_id": research_command_id,
            "research_attempt_id": research_run_id,
            "source_url": str(row.get("requested_url") or canonical_url),
            "verification_status_original": row.get("verification_status"),
            "verification_status": "verified",
            "transport_verified": True,
            "content_extracted": True,
            "actual_full_text_or_data": True,
            "evidence_relevance_score": relevance,
            "evidence_relevance_score_original": str(relevance_original),
            "evidence_relevance_scale_original": "external_audit_0_100",
            "http_status": http_status,
            "evidence_eligible": True,
            "evidence_rejection_reason": "",
            "review_status": "admitted",
            "read_status": "verified",
            "metadata_only": False,
            "snapshot_id": f"snapshot-{source_id.lower()}",
            "snapshot_path": target_path,
            "snapshot_hash": f"sha256:{expected_hash}",
            "evidence_id": evidence_id,
            "retrieved_at": str(row.get("checked_at") or ""),
            "url_role": "dataset_archive" if archive_verified else "original_content",
            "content_scope": "full_dataset" if archive_verified else "full_text",
            "source_receipt_id": f"receipt-{source_id.lower()}",
            "contribution_note": (
                "Original content read, snapshot retained, and SHA-256 verified "
                "by the independent baseline audit."
            ),
        }
    )
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-snapshot-root", required=True)
    parser.add_argument("--research-run-id", required=True)
    parser.add_argument("--research-command-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    normalized = [
        normalize_row(
            row,
            source_root=args.source_root,
            target_snapshot_root=args.target_snapshot_root,
            research_run_id=args.research_run_id,
            research_command_id=args.research_command_id,
        )
        for row in rows
    ]
    if not normalized:
        raise NormalizeError("empty_source_catalog")
    if len({row["source_id"] for row in normalized}) != len(normalized):
        raise NormalizeError("duplicate_source_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "rows": len(normalized), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

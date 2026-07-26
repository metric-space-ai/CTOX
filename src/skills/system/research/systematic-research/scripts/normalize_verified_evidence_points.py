#!/usr/bin/env python3
"""Normalize external-audit evidence against an admitted CTOX source catalog.

The independent SKF baseline uses a 0-100 relevance scale with decimal values,
while the knowledge import storage contract requires an integer. Preserve the
external score verbatim in an audit field and use conventional half-up rounding
for the indexed integer. This adapter does not alter the native 0-10
``ctox_web_read`` receipt contract used by CTOX-managed research.

All other evidence lineage remains fail-closed and is checked against the
already admitted source receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


class NormalizeError(ValueError):
    """The evidence row cannot be admitted fail-closed."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows", payload.get("data"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise NormalizeError(f"{path}:expected_json_row_array")
    return payload


def normalize_hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise NormalizeError("invalid_snapshot_hash")
    return text


def normalize_score(value: Any) -> tuple[int, str]:
    if isinstance(value, bool):
        raise NormalizeError("invalid_relevance_score")
    try:
        score = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise NormalizeError("invalid_relevance_score") from exc
    if not math.isfinite(float(score)):
        raise NormalizeError("non_finite_relevance_score")
    if not Decimal("8") <= score <= Decimal("100"):
        raise NormalizeError("relevance_score_out_of_range")
    return int(score.quantize(Decimal("1"), rounding=ROUND_HALF_UP)), str(score)


def normalize_rows(
    evidence_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources = {str(row.get("source_id") or ""): row for row in source_rows}
    if "" in sources:
        raise NormalizeError("source_catalog_missing_source_id")

    normalized: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()
    for index, row in enumerate(evidence_rows):
        source_id = str(row.get("source_id") or "")
        evidence_id = str(row.get("evidence_id") or "")
        prefix = evidence_id or f"row-{index}"
        source = sources.get(source_id)
        if source is None:
            raise NormalizeError(f"{prefix}:unknown_source_id:{source_id}")
        if not evidence_id:
            raise NormalizeError(f"{prefix}:missing_evidence_id")
        if evidence_id in seen_evidence_ids:
            raise NormalizeError(f"{prefix}:duplicate_evidence_id")
        seen_evidence_ids.add(evidence_id)

        for field in (
            "canonical_url",
            "snapshot_id",
            "snapshot_path",
            "source_receipt_id",
        ):
            if str(row.get(field) or "") != str(source.get(field) or ""):
                raise NormalizeError(f"{prefix}:{field}_source_mismatch")
        if normalize_hash(row.get("snapshot_hash")) != normalize_hash(
            source.get("snapshot_hash")
        ):
            raise NormalizeError(f"{prefix}:snapshot_hash_source_mismatch")
        if row.get("verification_status") != "verified":
            raise NormalizeError(f"{prefix}:verification_not_verified")
        for field in (
            "transport_verified",
            "content_extracted",
            "actual_full_text_or_data",
        ):
            if row.get(field) is not True:
                raise NormalizeError(f"{prefix}:{field}_not_true")
        status = row.get("http_status")
        if isinstance(status, bool) or not isinstance(status, int):
            raise NormalizeError(f"{prefix}:http_status_not_integer")
        if not 200 <= status <= 299 or status == 204:
            raise NormalizeError(f"{prefix}:http_status_not_usable")

        item = dict(row)
        normalized_score, original_score = normalize_score(
            row.get("evidence_relevance_score")
        )
        item["evidence_relevance_score"] = normalized_score
        item["evidence_relevance_score_original"] = original_score
        item["evidence_relevance_scale_original"] = "external_audit_0_100"
        item["evidence_eligible"] = True
        item["evidence_rejection_reason"] = ""
        normalized.append(item)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    normalized = normalize_rows(load_rows(args.input), load_rows(args.source_catalog))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "rows": len(normalized), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

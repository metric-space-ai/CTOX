#!/usr/bin/env python3
"""Separate legacy propeller derivations from unsupported bearing reactions."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OUTPUT_FIELDS = [
    "derivation_id",
    "evidence_id",
    "source_id",
    "measurement_id",
    "dataset_id",
    "source_file",
    "source_row_ref",
    "propeller_size_original",
    "prop_diameter_in",
    "prop_pitch_in",
    "diameter_m_input",
    "rpm_input",
    "n_rev_s",
    "air_density_kg_m3_input",
    "advance_ratio_J",
    "thrust_coefficient_CT_input",
    "power_coefficient_CP_input",
    "thrust_N_derived",
    "shaft_power_W_derived",
    "torque_Nm_derived",
    "formula_thrust",
    "formula_power",
    "formula_torque",
    "input_units",
    "result_units",
    "assumptions",
    "uncertainty",
    "confidence",
    "derivation_method",
    "is_derived",
    "canonical_url",
    "snapshot_sha256",
    "source_url",
    "snapshot_id",
    "snapshot_path",
    "snapshot_hash",
    "source_receipt_id",
    "retrieved_at",
    "url_role",
    "content_scope",
    "verification_status",
    "transport_verified",
    "content_extracted",
    "actual_full_text_or_data",
    "evidence_relevance_score",
    "http_status",
    "evidence_eligible",
    "source_tier",
    "source_type",
    "research_run_id",
    "research_command_id",
    "research_attempt_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", required=True, type=Path)
    parser.add_argument("--legacy-derived", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{row.get('derivation_id', 'row')} has no {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{row.get('derivation_id', 'row')} has invalid {field}")
    return number


def main() -> None:
    args = parse_args()
    direct_by_id = {
        row["measurement_id"].strip(): row
        for row in read_rows(args.direct)
        if row.get("measurement_id", "").strip()
    }
    legacy_rows = read_rows(args.legacy_derived)
    output_rows: list[dict[str, str]] = []

    for legacy in legacy_rows:
        measurement_id = legacy.get("measurement_id", "").strip()
        direct = direct_by_id.get(measurement_id)
        if direct is None:
            raise ValueError(f"missing direct measurement lineage for {measurement_id}")
        for radial_field in ("bearing_A_radial_load_N", "bearing_B_radial_load_N"):
            if legacy.get(radial_field, "").strip():
                raise ValueError(f"{legacy['derivation_id']} contains unsupported {radial_field}")

        rpm = as_float(legacy, "input_rpm")
        shaft_power = as_float(legacy, "derived_shaft_power_W")
        torque = as_float(legacy, "derived_torque_Nm")
        expected_torque = shaft_power / (2.0 * math.pi * (rpm / 60.0))
        if not math.isclose(torque, expected_torque, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{legacy['derivation_id']} fails the 2*pi torque audit")
        air_density = legacy.get("input_rho_kg_m3", "").strip()
        assumptions = (
            f"rho={air_density} kg/m3; static propeller coefficient scaling; "
            "nominal source propeller diameter; no bearing reaction inferred"
        )

        output_rows.append(
            {
                "derivation_id": legacy.get("derivation_id", ""),
                "evidence_id": legacy.get("evidence_id", ""),
                "source_id": legacy.get("source_id", ""),
                "measurement_id": measurement_id,
                "dataset_id": direct.get("dataset_id", ""),
                "source_file": legacy.get("source_file", ""),
                "source_row_ref": legacy.get("source_row_ref", ""),
                "propeller_size_original": direct.get("propeller_size_original", ""),
                "prop_diameter_in": direct.get("prop_diameter_in", ""),
                "prop_pitch_in": direct.get("prop_pitch_in", ""),
                "diameter_m_input": legacy.get("input_D_m", ""),
                "rpm_input": legacy.get("input_rpm", ""),
                "n_rev_s": f"{rpm / 60.0:.12g}",
                "air_density_kg_m3_input": legacy.get("input_rho_kg_m3", ""),
                "advance_ratio_J": direct.get("advance_ratio_J", ""),
                "thrust_coefficient_CT_input": legacy.get("input_CT", ""),
                "power_coefficient_CP_input": legacy.get("input_CP", ""),
                "thrust_N_derived": legacy.get("derived_thrust_N", ""),
                "shaft_power_W_derived": legacy.get("derived_shaft_power_W", ""),
                "torque_Nm_derived": legacy.get("derived_torque_Nm", ""),
                "formula_thrust": "T = CT * rho * n^2 * D^4",
                "formula_power": "P = CP * rho * n^3 * D^5",
                "formula_torque": "Q = P / (2*pi*n)",
                "input_units": "rpm=rev/min; n=rev/s; rho=kg/m3; D=m; CT=1; CP=1",
                "result_units": "T=N; P=W; Q=N*m",
                "assumptions": assumptions,
                "uncertainty": legacy.get("uncertainty", ""),
                "confidence": legacy.get("confidence", ""),
                "derivation_method": "coefficient_scaling",
                "is_derived": "true",
                **{
                    field: legacy.get(field, "")
                    for field in OUTPUT_FIELDS
                    if field
                    in {
                        "canonical_url",
                        "snapshot_sha256",
                        "source_url",
                        "snapshot_id",
                        "snapshot_path",
                        "snapshot_hash",
                        "source_receipt_id",
                        "retrieved_at",
                        "url_role",
                        "content_scope",
                        "verification_status",
                        "transport_verified",
                        "content_extracted",
                        "actual_full_text_or_data",
                        "evidence_relevance_score",
                        "http_status",
                        "evidence_eligible",
                        "source_tier",
                        "source_type",
                        "research_run_id",
                        "research_command_id",
                        "research_attempt_id",
                    }
                },
            }
        )

    if len(output_rows) != len(legacy_rows):
        raise ValueError("row-count mismatch during migration")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"validated_rows={len(output_rows)}")


if __name__ == "__main__":
    main()

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("normalize_verified_evidence_points.py")
SPEC = importlib.util.spec_from_file_location("normalize_verified_evidence_points", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NormalizeVerifiedEvidencePointsTests(unittest.TestCase):
    def source(self):
        return {
            "source_id": "SRC-0001",
            "canonical_url": "https://example.test/paper.pdf",
            "snapshot_id": "snapshot-src-0001",
            "snapshot_path": "/snapshots/paper.pdf",
            "snapshot_hash": "sha256:" + "a" * 64,
            "source_receipt_id": "receipt-src-0001",
        }

    def evidence(self):
        return {
            **self.source(),
            "evidence_id": "EVID-0001",
            "claim_id": "CLM-0001",
            "verification_status": "verified",
            "transport_verified": True,
            "content_extracted": True,
            "actual_full_text_or_data": True,
            "http_status": 200,
            "evidence_relevance_score": 81.8,
            "evidence_eligible": False,
            "evidence_rejection_reason": "evidence_relevance_below_threshold",
        }

    def test_preserves_and_rounds_external_audit_score(self):
        rows = MODULE.normalize_rows([self.evidence()], [self.source()])
        self.assertEqual(rows[0]["evidence_relevance_score"], 82)
        self.assertEqual(rows[0]["evidence_relevance_score_original"], "81.8")
        self.assertEqual(
            rows[0]["evidence_relevance_scale_original"], "external_audit_0_100"
        )
        self.assertIs(rows[0]["evidence_eligible"], True)
        self.assertEqual(rows[0]["evidence_rejection_reason"], "")

    def test_accepts_fractional_score_and_rejects_out_of_range_score(self):
        row = self.evidence()
        row["evidence_relevance_score"] = 9.5
        self.assertEqual(
            MODULE.normalize_rows([row], [self.source()])[0][
                "evidence_relevance_score"
            ],
            10,
        )
        for score in (7, 101):
            row = self.evidence()
            row["evidence_relevance_score"] = score
            with self.assertRaises(MODULE.NormalizeError):
                MODULE.normalize_rows([row], [self.source()])

    def test_rejects_lineage_mismatch(self):
        row = self.evidence()
        row["snapshot_hash"] = "b" * 64
        with self.assertRaisesRegex(
            MODULE.NormalizeError, "snapshot_hash_source_mismatch"
        ):
            MODULE.normalize_rows([row], [self.source()])

    def test_rejects_duplicate_evidence_id(self):
        row = self.evidence()
        with self.assertRaisesRegex(MODULE.NormalizeError, "duplicate_evidence_id"):
            MODULE.normalize_rows([row, row], [self.source()])


if __name__ == "__main__":
    unittest.main()

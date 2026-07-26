from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import normalize_verified_source_catalog as normalizer  # noqa: E402


class NormalizeVerifiedSourceCatalogTests(unittest.TestCase):
    def test_verified_snapshot_is_translated_after_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshots" / "source.pdf"
            snapshot.parent.mkdir()
            snapshot.write_bytes(b"verified original")
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            row = normalizer.normalize_row(
                {
                    "source_id": "SRC-1",
                    "canonical_url": "https://example.test/source.pdf",
                    "requested_url": "https://example.test/source",
                    "http_status": "200",
                    "evidence_relevance_score": "9",
                    "verification_status": "verified_original_read",
                    "evidence_eligible": "true",
                    "rejection_reason": "",
                    "snapshot_path": "snapshots/source.pdf",
                    "snapshot_sha256": digest,
                },
                source_root=root,
                target_snapshot_root="/srv/research",
                research_run_id="run-1",
                research_command_id="command-1",
            )
            self.assertEqual(row["verification_status"], "verified")
            self.assertEqual(row["verification_status_original"], "verified_original_read")
            self.assertEqual(row["snapshot_hash"], f"sha256:{digest}")
            self.assertTrue(row["evidence_eligible"])
            self.assertEqual(row["content_scope"], "full_text")

    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "source.pdf"
            snapshot.write_bytes(b"tampered")
            with self.assertRaisesRegex(normalizer.NormalizeError, "snapshot_hash_mismatch"):
                normalizer.normalize_row(
                    {
                        "source_id": "SRC-1",
                        "canonical_url": "https://example.test/source.pdf",
                        "http_status": "200",
                        "evidence_relevance_score": "9",
                        "verification_status": "verified_original_read",
                        "evidence_eligible": "true",
                        "snapshot_path": "source.pdf",
                        "snapshot_sha256": "a" * 64,
                    },
                    source_root=root,
                    target_snapshot_root="/srv/research",
                    research_run_id="run-1",
                    research_command_id="command-1",
                )


if __name__ == "__main__":
    unittest.main()

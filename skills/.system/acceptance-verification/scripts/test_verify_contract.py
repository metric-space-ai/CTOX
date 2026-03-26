#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "verify_contract.py"


class VerifyContractTests(unittest.TestCase):
    def run_script(self, payload):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            temp_path = handle.name
        completed = subprocess.run(
            ["python3", str(SCRIPT), "--checks-json", temp_path],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_returns_needs_repair_when_higher_layer_fails(self):
        payload = [
            {"layer": "service_process", "ok": True},
            {"layer": "listener", "ok": True},
            {"layer": "http", "ok": True},
            {"layer": "authenticated_api", "ok": False, "cause": "secret_invalid", "detail": "401"},
        ]
        result = self.run_script(payload)
        self.assertEqual(result["state"], "needs_repair")
        self.assertEqual(result["failed_layer"]["layer"], "authenticated_api")
        self.assertEqual(result["failed_layer"]["cause"], "secret_invalid")

    def test_returns_executed_when_all_known_layers_pass(self):
        payload = [
            {"layer": "service_process", "ok": True},
            {"layer": "listener", "ok": True},
            {"layer": "http", "ok": True},
            {"layer": "authenticated_api", "ok": True},
            {"layer": "admin_identity", "ok": True},
        ]
        result = self.run_script(payload)
        self.assertEqual(result["state"], "executed")
        self.assertIsNone(result["failed_layer"])


if __name__ == "__main__":
    unittest.main()

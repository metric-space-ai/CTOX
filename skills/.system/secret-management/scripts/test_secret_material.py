#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "secret_material.py"


class SecretMaterialTest(unittest.TestCase):
    def test_upsert_env_and_describe(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "runtime" / "secrets" / "demo.env"
            result = subprocess.check_output(
                [
                    "python3",
                    str(SCRIPT),
                    "upsert-env",
                    "--path",
                    str(target),
                    "--set",
                    "DEMO_USER=admin",
                    "--set",
                    "DEMO_PASSWORD=secret",
                ],
                text=True,
            )
            payload = json.loads(result)
            self.assertTrue(payload["exists"])
            self.assertEqual(payload["keys"], ["DEMO_PASSWORD", "DEMO_USER"])
            self.assertEqual(oct(target.stat().st_mode & 0o777), "0o600")


if __name__ == "__main__":
    unittest.main()

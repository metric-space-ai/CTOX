from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/configure_model_policy.py"


class ConfigureModelPolicyTests(unittest.TestCase):
    def run_script(self, profile: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "model-policy.json"
            policy_path.write_text('{"version":1,"grosshirnCandidates":[]}\n', encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--policy", str(policy_path), "--profile", profile],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn('"status": "ok"', result.stdout)
            return json.loads(policy_path.read_text(encoding="utf-8"))

    def test_sets_gpt_oss_profile(self) -> None:
        policy = self.run_script("gpt_oss")
        self.assertEqual(policy["kleinhirn"]["modelId"], "gpt-oss-20b")
        self.assertEqual(policy["kleinhirn"]["runtimeModelId"], "openai/gpt-oss-20b")
        self.assertEqual(policy["kleinhirn"]["agenticAdapter"], "openai_compatible_chat")
        self.assertEqual(policy["kleinhirn"]["startupMaxSeqs"], 1)
        self.assertEqual(policy["kleinhirn"]["startupMaxBatchSize"], 1)
        self.assertEqual(policy["kleinhirn"]["startupMaxSeqLen"], 8192)
        self.assertEqual(policy["kleinhirn"]["startupPagedAttnMode"], "off")
        self.assertEqual(policy["kleinhirnInstallAlternatives"][0]["modelId"], "Qwen3.5-35B-A3B")
        self.assertTrue(policy["kleinhirnInstallAlternatives"][0]["supportsVision"])
        self.assertEqual(policy["kleinhirnInstallAlternatives"][0]["startupPaCacheType"], "f8e4m3")

    def test_sets_qwen35_profile(self) -> None:
        policy = self.run_script("qwen35")
        self.assertEqual(policy["kleinhirn"]["modelId"], "Qwen3.5-0.8B")
        self.assertEqual(policy["kleinhirn"]["runtimeModelId"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(policy["kleinhirn"]["agenticAdapter"], "openai_compatible_chat")
        self.assertEqual(policy["kleinhirn"]["startupMaxSeqs"], 1)
        self.assertEqual(policy["kleinhirn"]["startupMaxBatchSize"], 1)
        self.assertEqual(policy["kleinhirn"]["startupMaxSeqLen"], 8192)
        self.assertEqual(policy["kleinhirn"]["startupPaContextLen"], 4096)
        self.assertEqual(policy["kleinhirn"]["startupPaCacheType"], "f8e4m3")
        self.assertEqual(policy["kleinhirn"]["startupPagedAttnMode"], "auto")
        self.assertTrue(policy["kleinhirn"]["preferAutoDeviceMapping"])
        self.assertFalse(policy["kleinhirn"]["supportsVision"])
        self.assertEqual(
            [item["modelId"] for item in policy["kleinhirnInstallAlternatives"]],
            ["Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-35B-A3B"],
        )
        self.assertTrue(policy["kleinhirnInstallAlternatives"][-1]["supportsVision"])


if __name__ == "__main__":
    unittest.main()

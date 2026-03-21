#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


GPT_OSS_20B = {
    "role": "kleinhirn",
    "provider": "openai",
    "modelId": "gpt-oss-20b",
    "runtimeModelId": "openai/gpt-oss-20b",
    "officialLabel": "GPT-OSS 20B",
    "agenticAdapter": "mistralrs_gpt_oss_harmony_completion",
    "reasoningEffort": "high",
    "deploymentMode": "local_or_self_hosted",
    "purpose": "always-on low-latency control, bootstrap discipline, supervision, prioritization, summaries",
    "minCpuThreads": 8,
    "minMemoryGb": 16,
    "minGpuCount": 1,
    "minTotalGpuMemoryGb": 12,
    "minSingleGpuMemoryGb": 12,
    "startupMaxSeqs": 1,
    "startupMaxBatchSize": 1,
    "startupMaxSeqLen": 131072,
    "startupPagedAttnMode": "off",
    "startupMultiGpuMode": "auto_device_map",
    "startupTensorParallelBackend": "disabled",
    "startupVisibleGpuPolicy": "all",
}

QWEN35_0_8B = {
    "role": "kleinhirn",
    "provider": "qwen",
    "modelId": "Qwen3.5-0.8B",
    "runtimeModelId": "Qwen/Qwen3.5-0.8B",
    "officialLabel": "Qwen3.5 0.8B",
    "agenticAdapter": "openai_compatible_chat",
    "reasoningEffort": "high",
    "deploymentMode": "local_or_self_hosted",
    "purpose": "smallest local Qwen3.5 always-on supervisor when the host cannot yet carry the larger family members",
    "supportsVision": False,
    "minCpuThreads": 4,
    "minMemoryGb": 8,
    "minGpuCount": 1,
    "minTotalGpuMemoryGb": 4,
    "minSingleGpuMemoryGb": 4,
    "startupMaxSeqs": 1,
    "startupMaxBatchSize": 1,
    "startupMaxSeqLen": 8192,
    "startupPaContextLen": 4096,
    "startupPaCacheType": "f8e4m3",
    "startupPagedAttnMode": "auto",
    "startupMultiGpuMode": "tensor_parallel",
    "startupTensorParallelBackend": "nccl",
    "startupVisibleGpuPolicy": "largest_power_of_two_prefer_display_free",
    "preferAutoDeviceMapping": False,
}

QWEN35_2B = {
    "role": "kleinhirn_install_alternative",
    "provider": "qwen",
    "modelId": "Qwen3.5-2B",
    "runtimeModelId": "Qwen/Qwen3.5-2B",
    "officialLabel": "Qwen3.5 2B",
    "agenticAdapter": "openai_compatible_chat",
    "reasoningEffort": "high",
    "deploymentMode": "local_or_self_hosted",
    "purpose": "small local Qwen3.5 supervisor when the host can carry a little more context and quality than 0.8B",
    "supportsVision": False,
    "minCpuThreads": 8,
    "minMemoryGb": 12,
    "minGpuCount": 1,
    "minTotalGpuMemoryGb": 6,
    "minSingleGpuMemoryGb": 6,
    "startupMaxSeqs": 1,
    "startupMaxBatchSize": 1,
    "startupMaxSeqLen": 8192,
    "startupPaContextLen": 6144,
    "startupPaCacheType": "f8e4m3",
    "startupPagedAttnMode": "auto",
    "startupMultiGpuMode": "tensor_parallel",
    "startupTensorParallelBackend": "nccl",
    "startupVisibleGpuPolicy": "largest_power_of_two_prefer_display_free",
    "preferAutoDeviceMapping": False,
}

QWEN35_4B = {
    "role": "kleinhirn_install_alternative",
    "provider": "qwen",
    "modelId": "Qwen3.5-4B",
    "runtimeModelId": "Qwen/Qwen3.5-4B",
    "officialLabel": "Qwen3.5 4B",
    "agenticAdapter": "openai_compatible_chat",
    "reasoningEffort": "high",
    "deploymentMode": "local_or_self_hosted",
    "purpose": "mid-sized local Qwen3.5 supervisor when the host can carry a materially stronger local family member without needing the 35B tier yet",
    "supportsVision": False,
    "minCpuThreads": 8,
    "minMemoryGb": 16,
    "minGpuCount": 1,
    "minTotalGpuMemoryGb": 12,
    "minSingleGpuMemoryGb": 12,
    "startupMaxSeqs": 1,
    "startupMaxBatchSize": 1,
    "startupMaxSeqLen": 131072,
    "startupPaContextLen": 131072,
    "startupPaCacheType": "f8e4m3",
    "startupPagedAttnMode": "auto",
    "startupMultiGpuMode": "tensor_parallel",
    "startupTensorParallelBackend": "nccl",
    "startupVisibleGpuPolicy": "largest_power_of_two_prefer_display_free",
    "preferAutoDeviceMapping": False,
}

INSTALL_ALTERNATIVES = []

QWEN35_FAMILY = [
    QWEN35_0_8B,
    QWEN35_2B,
    QWEN35_4B,
    QWEN35_35B_A3B,
]

UPGRADE_CANDIDATES = []

GROSSHIRN_CANDIDATES = [
    {
        "role": "grosshirn_candidate",
        "provider": "openai",
        "modelId": "gpt-5.4-nano",
        "runtimeModelId": "openai/gpt-5.4-nano",
        "officialLabel": "GPT-5.4 Nano",
        "agenticAdapter": "openai_responses",
        "reasoningEffort": "low",
        "deploymentMode": "external_api",
        "purpose": "small external model for the lightest approved grosshirn routing path",
        "supportsVision": True,
    },
    {
        "role": "grosshirn_candidate",
        "provider": "openai",
        "modelId": "gpt-5.4-mini",
        "runtimeModelId": "openai/gpt-5.4-mini",
        "officialLabel": "GPT-5.4 Mini",
        "agenticAdapter": "openai_responses",
        "reasoningEffort": "medium",
        "deploymentMode": "external_api",
        "purpose": "approved medium external model for heavier reasoning when GPT-OSS 20B is no longer sufficient",
        "supportsVision": True,
    },
    {
        "role": "grosshirn_candidate",
        "provider": "openai",
        "modelId": "gpt-5.4",
        "runtimeModelId": "openai/gpt-5.4",
        "officialLabel": "GPT-5.4",
        "agenticAdapter": "openai_responses",
        "reasoningEffort": "high",
        "deploymentMode": "external_api",
        "purpose": "external grosshirn for hard coding, agentic reasoning and complex task recovery when local kleinhirn is insufficient",
        "supportsVision": True,
    },
]


def normalize_profile_name(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {
        "qwen3",
        "qwen",
        "qwen3-30b-a3b",
        "qwen/qwen3-30b-a3b",
        "qwen35",
        "qwen3.5",
        "qwen3.5-35b-a3b",
        "qwen35-35b-a3b",
    }:
        return "qwen35"
    return "gpt_oss"


def load_policy(path: Path) -> dict:
    if not path.exists():
        return {"version": 1}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {"version": 1}
    return json.loads(raw)


def save_policy(path: Path, policy: dict) -> None:
    path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    policy_path = Path(args.policy).resolve()
    policy = load_policy(policy_path)

    profile = normalize_profile_name(args.profile)
    if profile == "qwen35":
        selected = deepcopy(QWEN35_35B_A3B)
        install_alternatives = deepcopy(QWEN35_FAMILY[:-1])
    else:
        selected = deepcopy(GPT_OSS_20B)
        install_alternatives = deepcopy(INSTALL_ALTERNATIVES)

    policy["version"] = int(policy.get("version") or 1)
    policy["kleinhirn"] = selected
    policy["kleinhirnInstallAlternatives"] = install_alternatives
    policy["kleinhirnUpgradeAllowed"] = True
    policy["kleinhirnUpgradeIndependentFromGrosshirn"] = True
    policy["kleinhirnUpgradeCandidates"] = deepcopy(UPGRADE_CANDIDATES)
    policy["grosshirnCandidates"] = deepcopy(GROSSHIRN_CANDIDATES)
    policy["updatedAt"] = now_iso()

    save_policy(policy_path, policy)
    print(json.dumps({"status": "ok", "profile": profile, "kleinhirn": selected["officialLabel"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

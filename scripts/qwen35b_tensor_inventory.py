#!/usr/bin/env python3

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def find_snapshot_dir(start: Path) -> Path:
    if start.is_dir() and any(start.glob("model.safetensors-*.safetensors")):
        return start

    candidates = []
    for path in start.rglob("*"):
        if path.is_dir() and "Qwen3.5-35B-A3B" in str(path):
            if any(path.glob("model.safetensors-*.safetensors")):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No safetensor shard directory for Qwen3.5-35B-A3B found")
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def classify_tensor(name: str) -> str:
    if name.startswith("model.language_model.embed_tokens."):
        return "embed_tokens"
    if name.startswith("model.language_model.norm."):
        return "final_norm"
    if name.startswith("lm_head."):
        return "lm_head"
    if name.startswith("visual.") or name.startswith("model.visual."):
        return "vision"

    layer_match = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)", name)
    if not layer_match:
        return "other"

    suffix = layer_match.group(2)
    if suffix.startswith("self_attn."):
        return "layer.self_attn"
    if suffix.startswith("mlp.gate."):
        return "layer.moe_gate"
    if suffix.startswith("mlp.shared_expert."):
        return "layer.shared_expert"
    if suffix.startswith("mlp.shared_expert_gate."):
        return "layer.shared_expert_gate"
    if suffix.startswith("mlp.experts."):
        return "layer.moe_experts"
    if suffix.startswith("input_layernorm.") or suffix.startswith("post_attention_layernorm."):
        return "layer.norm"
    return "layer.other"


def layer_group(name: str) -> str | None:
    match = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)", name)
    if not match:
        return None
    idx = int(match.group(1))
    suffix = match.group(2)
    if suffix.startswith("self_attn."):
        return f"layer.{idx:02d}.self_attn"
    if suffix.startswith("mlp.gate."):
        return f"layer.{idx:02d}.moe_gate"
    if suffix.startswith("mlp.shared_expert."):
        return f"layer.{idx:02d}.shared_expert"
    if suffix.startswith("mlp.shared_expert_gate."):
        return f"layer.{idx:02d}.shared_expert_gate"
    if suffix.startswith("mlp.experts."):
        return f"layer.{idx:02d}.moe_experts"
    if suffix.startswith("input_layernorm.") or suffix.startswith("post_attention_layernorm."):
        return f"layer.{idx:02d}.norm"
    return f"layer.{idx:02d}.other"


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("HF_HOME", Path.home()))
    snapshot_dir = find_snapshot_dir(root)
    shard_paths = sorted(snapshot_dir.glob("model.safetensors-*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No model.safetensors shards found in {snapshot_dir}")

    weight_map = {}
    total_size = 0
    shard_sizes: dict[str, int] = defaultdict(int)
    tensor_bytes: dict[str, int] = {}
    for shard_path in shard_paths:
        shard = shard_path.name
        with shard_path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len))
        for tensor_name, desc in header.items():
            if tensor_name == "__metadata__":
                continue
            offsets = desc.get("data_offsets")
            if offsets and len(offsets) == 2:
                nbytes = int(offsets[1]) - int(offsets[0])
                tensor_bytes[tensor_name] = nbytes
                weight_map[tensor_name] = shard
                shard_sizes[shard] += nbytes
                total_size += nbytes

    class_byte_sizes: dict[str, int] = defaultdict(int)
    layer_byte_sizes: dict[str, int] = defaultdict(int)
    top_tensors = []
    top_global_tensors = []
    for name, nbytes in tensor_bytes.items():
        class_byte_sizes[classify_tensor(name)] += nbytes
        layer_key = layer_group(name)
        if layer_key:
            layer_byte_sizes[layer_key] += nbytes
        else:
            top_global_tensors.append((nbytes, name))
        top_tensors.append((nbytes, name))

    top_tensors.sort(reverse=True)
    top_global_tensors.sort(reverse=True)

    result = {
        "snapshot_dir": str(snapshot_dir),
        "total_size_bytes": total_size,
        "discovered_total_tensor_bytes": sum(tensor_bytes.values()),
        "class_byte_sizes": dict(sorted(class_byte_sizes.items())),
        "layer_byte_sizes": dict(sorted(layer_byte_sizes.items())),
        "top_tensors": [
            {"name": name, "bytes": nbytes} for nbytes, name in top_tensors[:80]
        ],
        "top_global_tensors": [
            {"name": name, "bytes": nbytes}
            for nbytes, name in top_global_tensors[:80]
        ],
        "tensor_count": len(weight_map),
        "shard_sizes": dict(sorted(shard_sizes.items())),
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time


def module_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"MISSING: {exc!r}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nvidia/Nemotron-Cascade-2-30B-A3B",
        help="HF model id",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "<|im_start|>system\n"
            "You are a helpful assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Reply with exactly OK.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        help="Prompt to evaluate",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--hidden-stats",
        action="store_true",
        help="Emit compact per-layer hidden-state stats for the prompt forward.",
    )
    args = parser.parse_args()

    env_info = {
        "python": sys.version,
        "cwd": os.getcwd(),
        "modules": {
            "torch": module_version("torch"),
            "transformers": module_version("transformers"),
            "accelerate": module_version("accelerate"),
            "safetensors": module_version("safetensors"),
            "huggingface_hub": module_version("huggingface_hub"),
        },
    }
    print(json.dumps({"stage": "env", "info": env_info}, ensure_ascii=True), flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        cuda_info["devices"].append(
            {
                "index": idx,
                "name": props.name,
                "total_memory": props.total_memory,
            }
        )
    print(json.dumps({"stage": "cuda", "info": cuda_info}, ensure_ascii=True), flush=True)

    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    load_sec = time.time() - load_start
    print(
        json.dumps(
            {
                "stage": "load",
                "seconds": round(load_sec, 3),
                "hf_device_map": getattr(model, "hf_device_map", None),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    tok_start = time.time()
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    token_sec = time.time() - tok_start
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    model_device = model.get_input_embeddings().weight.device
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)
    input_ids = input_ids.to(model_device)

    with torch.no_grad():
        fwd_start = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=args.hidden_stats,
            return_dict=True,
        )
        fwd_sec = time.time() - fwd_start

    next_logits = outputs.logits[:, -1, :].float().cpu()
    probs = torch.softmax(next_logits, dim=-1)
    topk = min(args.topk, next_logits.shape[-1])
    top_probs, top_indices = probs.topk(topk, dim=-1)

    rows = []
    for rank in range(topk):
        token_id = int(top_indices[0, rank].item())
        rows.append(
            {
                "rank": rank + 1,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "prob": float(top_probs[0, rank].item()),
                "logit": float(next_logits[0, token_id].item()),
            }
        )

    best_token_id = int(top_indices[0, 0].item())
    best_token = tokenizer.decode([best_token_id])
    print(
        json.dumps(
            {
                "stage": "first_step",
                "tokenize_seconds": round(token_sec, 3),
                "forward_seconds": round(fwd_sec, 3),
                "prompt_tokens": int(input_ids.shape[-1]),
                "best_token_id": best_token_id,
                "best_token_text": best_token,
                "topk": rows,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    if args.hidden_stats:
        hidden_stats = []
        for idx, hidden in enumerate(outputs.hidden_states or ()):
            hidden_f32 = hidden.float().cpu()
            hidden_stats.append(
                {
                    "layer": idx - 1,
                    "shape": list(hidden_f32.shape),
                    "abs_mean": float(hidden_f32.abs().mean().item()),
                    "last_token_abs_mean": float(hidden_f32[:, -1, :].abs().mean().item()),
                }
            )
        print(
            json.dumps(
                {
                    "stage": "hidden_stats",
                    "layers": hidden_stats,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

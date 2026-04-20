"""
Benchmark inference latency and throughput before/after KV cache changes.

Typical usage:

python -m scripts.bench_infer \
    --source base \
    --model-tag d2 \
    --prompt-lens 128,256,512,1024 \
    --decode-len 128 \
    --measure-runs 5 \
    --label before_kvcache \
    --out runs/bench_before_kvcache.json

Run the same command again after your KV cache changes and compare the JSON files
with `python -m scripts.compare_bench`.
"""

import argparse
import gc
import json
import math
import os
import time
from contextlib import nullcontext
from statistics import mean, pstdev

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.engine import Engine, KVCache


def parse_int_list(csv_value):
    values = []
    for chunk in csv_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def maybe_reset_peak_memory(device):
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.memory_allocated(device)
    return None


def maybe_get_peak_memory_delta_mib(device, start_allocated):
    if device.type != "cuda" or start_allocated is None:
        return None
    peak = torch.cuda.max_memory_allocated(device)
    return (peak - start_allocated) / (1024 * 1024)


def build_prompt_tokens(prompt_len, bos_token_id, vocab_size):
    assert prompt_len >= 1, "prompt length must be at least 1"
    tokens = [bos_token_id]
    usable_vocab = max(2, min(vocab_size, 251))
    for i in range(prompt_len - 1):
        tokens.append(1 + (i % (usable_vocab - 1)))
    return tokens


def summarize_numeric(records, keys):
    summary = {}
    for key in keys:
        values = [record[key] for record in records if record[key] is not None]
        if not values:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
            continue
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = 0.0 if len(values) == 1 else pstdev(values)
    return summary


@torch.inference_mode()
def benchmark_stages_once(model, prompt_tokens, decode_len, device, autocast_ctx):
    m = model.config
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    kv_model_kwargs = {
        "num_heads": m.n_kv_head,
        "head_dim": m.n_embd // m.n_head,
        "num_layers": m.n_layer,
    }

    start_allocated = maybe_reset_peak_memory(device)
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    synchronize(device)
    t0 = time.perf_counter()
    kv_cache_prefill = KVCache(
        batch_size=1,
        seq_len=len(prompt_tokens),
        device=device,
        dtype=dtype,
        **kv_model_kwargs,
    )
    with autocast_ctx:
        logits = model.forward(ids, kv_cache=kv_cache_prefill)
    synchronize(device)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    kv_cache_decode = KVCache(
        batch_size=1,
        seq_len=len(prompt_tokens) + decode_len,
        device=device,
        dtype=dtype,
        **kv_model_kwargs,
    )
    kv_cache_decode.prefill(kv_cache_prefill)
    del kv_cache_prefill
    synchronize(device)
    t3 = time.perf_counter()

    next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    t4 = time.perf_counter()
    if decode_len > 0:
        with autocast_ctx:
            for _ in range(decode_len):
                logits = model.forward(next_ids, kv_cache=kv_cache_decode)
                next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    synchronize(device)
    t5 = time.perf_counter()

    peak_mem_delta_mib = maybe_get_peak_memory_delta_mib(device, start_allocated)

    prefill_s = t1 - t0
    clone_s = t3 - t2
    decode_s = t5 - t4
    total_s = prefill_s + clone_s + decode_s

    return {
        "prefill_ms": prefill_s * 1000,
        "clone_ms": clone_s * 1000,
        "decode_ms": decode_s * 1000,
        "total_ms": total_s * 1000,
        "prefill_tok_s": len(prompt_tokens) / prefill_s if prefill_s > 0 else None,
        "decode_tok_s": decode_len / decode_s if decode_len > 0 and decode_s > 0 else None,
        "gen_tok_s_e2e": decode_len / total_s if decode_len > 0 and total_s > 0 else None,
        "peak_mem_delta_mib": peak_mem_delta_mib,
    }


@torch.inference_mode()
def benchmark_engine_once(engine, prompt_tokens, decode_len, device, autocast_ctx):
    start_allocated = maybe_reset_peak_memory(device)
    synchronize(device)
    t0 = time.perf_counter()
    with autocast_ctx:
        engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=decode_len,
            temperature=0.0,
        )
    synchronize(device)
    t1 = time.perf_counter()

    total_s = t1 - t0
    peak_mem_delta_mib = maybe_get_peak_memory_delta_mib(device, start_allocated)
    total_tokens = len(prompt_tokens) + decode_len

    return {
        "engine_total_ms": total_s * 1000,
        "engine_gen_tok_s": decode_len / total_s if decode_len > 0 and total_s > 0 else None,
        "engine_total_tok_s": total_tokens / total_s if total_s > 0 else None,
        "peak_mem_delta_mib": peak_mem_delta_mib,
    }


def print_stage_table(results):
    header = (
        f"{'prompt':>8} {'decode':>8} {'prefill_ms':>12} {'clone_ms':>10} "
        f"{'decode_ms':>12} {'total_ms':>11} {'prefill_t/s':>13} "
        f"{'decode_t/s':>12} {'peak_mem(MiB)':>14}"
    )
    print0(header)
    print0("-" * len(header))
    for row in results:
        print0(
            f"{row['prompt_len']:8d} {row['decode_len']:8d} "
            f"{row['prefill_ms_mean']:12.2f} {row['clone_ms_mean']:10.2f} "
            f"{row['decode_ms_mean']:12.2f} {row['total_ms_mean']:11.2f} "
            f"{row['prefill_tok_s_mean']:13.2f} "
            f"{(row['decode_tok_s_mean'] if row['decode_tok_s_mean'] is not None else float('nan')):12.2f} "
            f"{(row['peak_mem_delta_mib_mean'] if row['peak_mem_delta_mib_mean'] is not None else float('nan')):14.2f}"
        )


def print_engine_table(results):
    header = (
        f"{'prompt':>8} {'decode':>8} {'engine_ms':>12} "
        f"{'gen_t/s':>12} {'total_t/s':>12} {'peak_mem(MiB)':>14}"
    )
    print0(header)
    print0("-" * len(header))
    for row in results:
        print0(
            f"{row['prompt_len']:8d} {row['decode_len']:8d} "
            f"{row['engine_total_ms_mean']:12.2f} "
            f"{(row['engine_gen_tok_s_mean'] if row['engine_gen_tok_s_mean'] is not None else float('nan')):12.2f} "
            f"{(row['engine_total_tok_s_mean'] if row['engine_total_tok_s_mean'] is not None else float('nan')):12.2f} "
            f"{(row['peak_mem_delta_mib_mean'] if row['peak_mem_delta_mib_mean'] is not None else float('nan')):14.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference before/after KV cache changes")
    parser.add_argument("--source", type=str, default="base", choices=["base", "sft", "rl"])
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"])
    parser.add_argument("--prompt-lens", type=str, default="128,256,512,1024")
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=5)
    parser.add_argument("--mode", type=str, default="stages", choices=["stages", "engine"])
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    prompt_lens = parse_int_list(args.prompt_lens)
    assert args.decode_len >= 0, "decode length must be non-negative"
    assert args.warmup_runs >= 0 and args.measure_runs >= 1, "invalid warmup/measure runs"

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    assert not ddp, "This benchmark is intended for single-process inference"

    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    max_prompt = max(prompt_lens)
    if max_prompt > model.config.sequence_len:
        raise ValueError(
            f"Requested prompt length {max_prompt} exceeds model sequence_len {model.config.sequence_len}. "
            "Keep the benchmark within the trained context size for a fair comparison."
        )
    if max_prompt + args.decode_len > model.rotary_seq_len:
        raise ValueError(
            f"Requested prompt+decode length {max_prompt + args.decode_len} exceeds rotary cache {model.rotary_seq_len}."
        )

    checkpoint_step = meta.get("step")
    model_tag = args.model_tag if args.model_tag is not None else "auto"
    bos_token_id = tokenizer.get_bos_token_id()

    print0(
        f"Benchmarking {args.source} model_tag={model_tag} step={checkpoint_step} "
        f"on {device.type} in {args.mode} mode"
    )

    results = []
    for prompt_len in prompt_lens:
        prompt_tokens = build_prompt_tokens(prompt_len, bos_token_id, tokenizer.get_vocab_size())

        warmup_fn = benchmark_stages_once if args.mode == "stages" else benchmark_engine_once
        for _ in range(args.warmup_runs):
            warmup_fn(model if args.mode == "stages" else engine, prompt_tokens, args.decode_len, device, autocast_ctx)

        records = []
        for _ in range(args.measure_runs):
            if args.mode == "stages":
                record = benchmark_stages_once(model, prompt_tokens, args.decode_len, device, autocast_ctx)
                keys = [
                    "prefill_ms",
                    "clone_ms",
                    "decode_ms",
                    "total_ms",
                    "prefill_tok_s",
                    "decode_tok_s",
                    "gen_tok_s_e2e",
                    "peak_mem_delta_mib",
                ]
            else:
                record = benchmark_engine_once(engine, prompt_tokens, args.decode_len, device, autocast_ctx)
                keys = [
                    "engine_total_ms",
                    "engine_gen_tok_s",
                    "engine_total_tok_s",
                    "peak_mem_delta_mib",
                ]
            records.append(record)

        summary = {
            "prompt_len": prompt_len,
            "decode_len": args.decode_len,
            "mode": args.mode,
            **summarize_numeric(records, keys),
        }
        results.append(summary)

    if args.mode == "stages":
        print_stage_table(results)
    else:
        print_engine_table(results)

    output = {
        "label": args.label,
        "source": args.source,
        "model_tag": model_tag,
        "checkpoint_step": checkpoint_step,
        "device_type": device.type,
        "mode": args.mode,
        "prompt_lens": prompt_lens,
        "decode_len": args.decode_len,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "results": results,
    }

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print0(f"Saved benchmark results to {args.out}")

    compute_cleanup()


if __name__ == "__main__":
    main()

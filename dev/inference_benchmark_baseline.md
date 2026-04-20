# Inference Benchmark Baseline

This document records the pre-optimization inference benchmark baseline for the `d12` base model. The goal is to establish a quantitative reference before implementing KV cache optimizations.

## Benchmark Setup

- Model tag: `d12`
- Checkpoint step: `352`
- Device: `cuda`
- Mode: `stages`
- Decode length: `128`
- Prompt lengths: `128`, `256`, `512`, `1024`

Command used:

```bash
python -m scripts.bench_infer \
  --source base \
  --model-tag d12 \
  --step 352 \
  --device-type cuda \
  --prompt-lens 128,256,512,1024 \
  --decode-len 128 \
  --warmup-runs 2 \
  --measure-runs 10 \
  --mode stages \
  --label before_kvcache \
  --out runs/bench_before_kvcache.json
```

## Raw Results

| prompt | decode | prefill_ms | clone_ms | decode_ms | total_ms | prefill_t/s | decode_t/s | peak_mem(MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 13.07 | 0.14 | 786.49 | 799.71 | 10890.95 | 162.75 | 147.82 |
| 256 | 128 | 14.65 | 0.14 | 796.52 | 811.31 | 21001.81 | 160.70 | 198.63 |
| 512 | 128 | 11.48 | 0.14 | 805.61 | 817.23 | 44588.80 | 158.89 | 299.13 |
| 1024 | 128 | 14.78 | 0.14 | 811.94 | 826.86 | 69291.94 | 157.65 | 496.51 |

## Key Observations

- Decode throughput remains relatively stable across prompt lengths, staying in the range of `157.65-162.75 tokens/s`.
- Per-token decode latency is roughly stable at about `6.1-6.3 ms/token`.
- Prefill latency is low for this `d12` setting and does not dominate end-to-end latency.
- Peak memory grows significantly with prompt length, from `147.82 MiB` at prompt length `128` to `496.51 MiB` at prompt length `1024`.

## Interpretation

- Current inference speed is already fairly stable during decode, so later KV cache work may not produce dramatic throughput gains in all cases.
- The more obvious optimization target is memory growth with longer contexts.
- Sliding-window KV cache management should therefore be evaluated primarily on:
  - reduced `peak_mem(MiB)`
  - stable or improved `decode_t/s`
  - stable or reduced `decode_ms`

## Project Description Draft

- Built an inference benchmark pipeline for the `d12` Decoder-only Transformer and profiled `prefill`, `decode`, and memory usage across multiple prompt lengths; with `decode_len=128`, generation throughput remained stable at `157.65-162.75 tokens/s` and per-token latency stayed near `6.3 ms`, while peak memory increased from `147.82 MiB` to `496.51 MiB`, establishing a quantitative baseline for subsequent sliding-window KV cache optimization.

## Next Step

Implement sliding-window KV cache trimming and rerun the same benchmark command to compare:

- `decode_t/s`
- `decode_ms`
- `peak_mem(MiB)`

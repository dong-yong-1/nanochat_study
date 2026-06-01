# SFT Run Log

## 2026-05-30 d12 Pretrain -> SFT

### Remote Context

- Remote project: `/root/autodl-tmp/nanochat`
- Original run base dir: `/root/autodl-tmp/nanochat_runs`
- Pretrain model tag: `d12_pretrain`
- Final base checkpoint: `/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_pretrain/model_002063.pt`
- GPU observed before SFT prep: NVIDIA GeForce RTX 4090 D, 24GB
- Network acceleration: `source /etc/network_turbo`

### Pretrain Summary

- Steps: `2063`
- Train tokens: `1,081,606,144`
- Total params: `203.7M`
- Scaling params: `103.0M`
- Token/scaling-param ratio: `10.50`
- Final validation BPB: `0.937980`
- Final CORE metric: `0.1116`
- Peak memory: `15220.97 MiB`
- Total training time: `195.46 min`

### SFT Prep Notes

- `/root/autodl-tmp` was full before SFT prep: `50G used, 191M available`.
- To avoid deleting existing artifacts, SFT outputs should be written under system disk path `/root/nanochat_runs_sft`.
- Planned writable base dir for SFT/eval: `/root/nanochat_runs_sft`
- Planned remote logs:
  - Base eval: `/root/nanochat_runs_sft/logs/base_eval_d12_pretrain.log`
  - SFT: `/root/nanochat_runs_sft/logs/d12_sft_live.log`

### Base Eval Command

```bash
cd /root/autodl-tmp/nanochat
source /etc/network_turbo
export NANOCHAT_BASE_DIR=/root/nanochat_runs_sft
export HF_HOME=/root/nanochat_runs_sft/hf_cache
export WANDB_MODE=offline
.venv/bin/python -m scripts.base_eval \
  --model-tag d12_pretrain \
  --eval core,bpb,sample \
  --device-batch-size 8 \
  --max-per-task 200 \
  --split-tokens 1048576
```

### SFT Command

```bash
cd /root/autodl-tmp/nanochat
source /etc/network_turbo
export NANOCHAT_BASE_DIR=/root/nanochat_runs_sft
export HF_HOME=/root/nanochat_runs_sft/hf_cache
export WANDB_MODE=offline
nohup env PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.chat_sft \
  --model-tag=d12_pretrain \
  --device-batch-size=8 \
  --total-batch-size=131072 \
  --num-iterations=1000 \
  --eval-every=200 \
  --eval-tokens=262144 \
  --chatcore-every=200 \
  --chatcore-max-cat=100 \
  --chatcore-max-sample=24 \
  --run=d12_sft \
  > /root/nanochat_runs_sft/logs/d12_sft_live.log 2>&1 < /dev/null &
```

### Status

- 2026-05-30: Connected to remote server and confirmed GPU is available.
- 2026-05-30: Detected data disk full; using `/root/nanochat_runs_sft` for new outputs to avoid deleting existing artifacts.
- 2026-05-30: User paused SFT before any SFT/base-eval job was launched. No SFT training has been started.
- 2026-05-31: Rechecked remote in no-GPU mode. The container cgroup memory limit is `2GB`, GPU is unavailable, and `/root/autodl-tmp` remains full (`50G/50G`, about `191M` available). This mode is not suitable for SFT.

### 2026-05-31 Resource Diagnosis

Current storage usage under `/root/autodl-tmp`:

| Path | Size | Note |
|---|---:|---|
| `/root/autodl-tmp/models` | 15G | Qwen model cache, unrelated to nanochat |
| `/root/autodl-tmp/MedicalGPT` | 12G | Another project, should not delete by default |
| `/root/autodl-tmp/nanochat_runs` | 12G | nanochat data/checkpoints |
| `/root/autodl-tmp/nanochat` | 7.3G | mostly `.venv` |
| `/root/autodl-tmp/hf-cache` | 3.7G | HF cache |
| `/root/autodl-tmp/.Trash-0` | 2.4G | trash folder |

nanochat-specific cleanup candidates:

| Candidate | Approx. space | Risk |
|---|---:|---|
| Delete `d12_gpu_smoke` and `d12_gpu_smoke_b8` checkpoints | 3.0G | Low; these are 2-step smoke runs |
| Keep only final `d12_pretrain` checkpoint and delete step 500/1000/1500/2000 model+optim files | 6.0G | Medium; loses intermediate checkpoint rollback, but final checkpoint remains |
| Empty `/root/autodl-tmp/.Trash-0` | 2.4G | Low/medium; depends on whether trash contents are needed |

Recommended SFT resource plan:

1. Run SFT only after GPU mode is enabled.
2. Free at least 6-10GB on `/root/autodl-tmp`, or route new SFT outputs/caches to `/root/nanochat_runs_sft` on the system disk.
3. Preserve `/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_pretrain/model_002063.pt`, `meta_002063.json`, and preferably `optim_002063_rank0.pt`.

Follow-up cleanup completed on 2026-05-31:

- Backed up MedicalGPT core files locally to `/Users/dongyong/Project/Trea_code/nanochat/remote_backups/MedicalGPT_20260531`.
- Deleted MedicalGPT remote `.venv`, `cache`, `outputs`, `outputs-eval`, and Python cache directories.
- `/root/autodl-tmp` now has about `12G` available.
- Detailed record: `dev/MEDICALGPT_REMOTE_CLEANUP_20260531.md`.

## 2026-05-31 SFT Launch

### Environment Check

- GPU mode was enabled: NVIDIA GeForce RTX 4090 D, 24GB.
- Remote memory mode was normal again: 18 CPU cores, 60GB RAM from the AutoDL banner.
- `/root/autodl-tmp` had about `12G` available before SFT prep.
- `identity_conversations.jsonl` was missing and was downloaded to `/root/autodl-tmp/nanochat_runs/identity_conversations.jsonl`.
- `source /etc/network_turbo` was applied.

### Smoke Run

Initial smoke command:

```bash
cd /root/autodl-tmp/nanochat
source /etc/network_turbo
export NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat_runs
export HF_HOME=/root/autodl-tmp/hf-cache
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
.venv/bin/python -m scripts.chat_sft \
  --model-tag=d12_pretrain \
  --device-batch-size=4 \
  --total-batch-size=131072 \
  --num-iterations=10 \
  --eval-every=10 \
  --eval-tokens=131072 \
  --chatcore-every=-1 \
  --run=d12_sft_smoke
```

The first smoke attempt failed before training because HuggingFace Hub attempted to use Xet/CAS for MMLU and returned `401 Unauthorized`.

The successful smoke used:

```bash
export HF_HUB_DISABLE_XET=1
unset OMP_NUM_THREADS
.venv/bin/python -m scripts.chat_sft \
  --model-tag=d12_pretrain \
  --device-batch-size=4 \
  --total-batch-size=131072 \
  --num-iterations=10 \
  --eval-every=10 \
  --eval-tokens=131072 \
  --chatcore-every=-1 \
  --run=d12_sft_smoke_xetoff
```

Smoke result:

| Metric | Value |
|---|---:|
| Initial validation BPB | `0.8655` |
| Completed optimizer steps | `1` |
| Peak memory | `8804.08 MiB` |
| Checkpoint save | successful |

Important script behavior found during smoke:

- In `scripts/chat_sft.py`, `--num-iterations` is checked inside the data generator, so it effectively counts dataloader yields / micro-batches, not optimizer steps.
- With `device_batch_size=4`, `total_batch_size=131072`, and `max_seq_len=2048`, gradient accumulation is `16`.
- Therefore, to run about `1000` optimizer steps, `--num-iterations` should be `16000`.

The smoke checkpoint files at step 1 were deleted after verification to free about `1.5G` for the formal SFT run.

### Formal Run

Launched at about `2026-05-31 07:58 CST`.

PID:

```text
4248
```

Command:

```bash
cd /root/autodl-tmp/nanochat
nohup env PYTHONUNBUFFERED=1 \
  NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat_runs \
  HF_HOME=/root/autodl-tmp/hf-cache \
  HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=offline \
  .venv/bin/python -m scripts.chat_sft \
  --model-tag=d12_pretrain \
  --device-batch-size=4 \
  --total-batch-size=131072 \
  --num-iterations=16000 \
  --eval-every=200 \
  --eval-tokens=262144 \
  --chatcore-every=200 \
  --chatcore-max-cat=100 \
  --chatcore-max-sample=24 \
  --run=d12_sft \
  > logs/d12_sft_live.log 2>&1 < /dev/null &
```

Log path:

```text
/root/autodl-tmp/nanochat/logs/d12_sft_live.log
```

W&B offline run:

```text
/root/autodl-tmp/nanochat/wandb/offline-run-20260531_075840-ybnjpfo3
```

Early formal-run status at `2026-05-31 08:00 CST`:

| Item | Value |
|---|---:|
| Step | `39` |
| Progress | `3.91%` |
| Initial validation BPB | `0.8348` |
| Latest train loss | `2.062148` |
| Throughput | about `88.5k tokens/sec` |
| MFU | about `40.7%` |
| Disk available after smoke cleanup | about `7.2G` |

Useful monitor commands:

```bash
cd /root/autodl-tmp/nanochat
ps -p 4248 -o pid,stat,etime,cmd
tail -f logs/d12_sft_live.log
df -h /root/autodl-tmp
```

### Formal Run Result

Checked on `2026-05-31`.

The formal SFT job completed successfully. The server was back in no-GPU mode when checked, but the log and final checkpoint were present.

Final artifacts:

| Artifact | Path |
|---|---|
| SFT log | `/root/autodl-tmp/nanochat/logs/d12_sft_live.log` |
| SFT checkpoint | `/root/autodl-tmp/nanochat_runs/chatsft_checkpoints/d12_pretrain/model_001000.pt` |
| SFT metadata | `/root/autodl-tmp/nanochat_runs/chatsft_checkpoints/d12_pretrain/meta_001000.json` |
| Optimizer state | `/root/autodl-tmp/nanochat_runs/chatsft_checkpoints/d12_pretrain/optim_001000_rank0.pt` |
| W&B offline run | `/root/autodl-tmp/nanochat/wandb/offline-run-20260531_075840-ybnjpfo3` |

Training result:

| Metric | Value |
|---|---:|
| Optimizer steps | `1000` |
| Initial SFT validation BPB | `0.8348` |
| Final/min SFT validation BPB | `0.5103` |
| Final train loss | `1.541367` |
| Peak memory | `8803.96 MiB` |
| Total training time | `24.42 min` |
| Throughput | about `88.5k tokens/sec` |
| MFU | about `40.7%` |

Final ChatCORE result:

| Metric | Value |
|---|---:|
| ChatCORE | `0.1706` |
| ChatCORE categorical | `0.0356` |
| ARC-Easy | `37/100 = 37.00%` |
| ARC-Challenge | `20/100 = 20.00%` |
| MMLU | `26/100 = 26.00%` |
| GSM8K | `0/24 = 0.00%` |
| HumanEval | `0/24 = 0.00%` |
| SpellingBee | `22/24 = 91.67%` |

Interpretation:

- The SFT pipeline is validated end-to-end: base checkpoint loading, mixed SFT data construction, gradient accumulation, validation, ChatCORE evaluation, checkpoint saving, and offline W&B logging all worked.
- SFT validation BPB improved from `0.8348` to `0.5103` within this run, so the model fit the SFT distribution substantially better.
- The strong SpellingBee score suggests the SFT stage successfully taught at least some format-following / exact-output behavior.
- GSM8K and HumanEval remained at `0`, so this run does not yet prove math/tool-use or code ability improvement.
- ARC/MMLU are still weak, which is expected for a small model trained for a short SFT run. This should be framed as a pipeline and targeted-ability validation, not as a strong general-capability result.

Resume-safe takeaway:

- It is safe to claim that the SFT stage was implemented and verified, with validation BPB decreasing from `0.8348` to `0.5103` and final ChatCORE reaching `0.1706`.
- It is not yet safe to claim that calculator/tool-use improved GSM8K performance. The next experiment should isolate a GSM8K/tool-use SFT set and compare before/after exact-match or execution-verified accuracy.

### Tool-Use Smoke Test

Checked on `2026-05-31` with GPU mode enabled.

Script:

```bash
python -m scripts.tool_use_smoke \
  --source=sft \
  --model-tag=d12_pretrain \
  --step=1000 \
  --device-type=cuda \
  --max-new-tokens=256 \
  --temperature=0.0 \
  --top-k=1 \
  --jsonl=logs/d12_sft_tool_use_smoke.jsonl
```

Local copy of per-prompt results:

```text
dev/d12_sft_tool_use_smoke.jsonl
```

Summary:

| Metric | Value |
|---|---:|
| Math prompts | `5` |
| Math tool-call rate | `0.00%` |
| Math tool-output rate | `0.00%` |
| Math answer accuracy | `0.00%` |
| Non-math tool calls | `0` |

Additional explicit prompt:

```text
Use the Python calculator tool to compute 347 * 28. Give the final answer after ####.
```

The model produced a markdown-style Python code block rather than the nanochat tool special tokens. It did not emit `<|python_start|>`, so the engine never invoked the calculator.

Diagnosis:

- The current SFT checkpoint does not yet learn the calculator tool-use protocol.
- GSM8K `0%` is primarily a tool-call triggering / special-token generation problem, not a calculator execution problem.
- The next SFT stage should start with calculator warmup data that densely supervises `<|python_start|> expression <|python_end|>` generation before moving to harder multi-step GSM8K.

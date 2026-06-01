# Remote Checkpoint Cleanup 2026-06-01

## Context

Server was in no-GPU mode:

| Item | Value |
|---|---|
| CPU | `0.5` core |
| Memory | `2GB` |
| GPU | none |
| Data disk before cleanup | `48G/50G`, `2.9G` available, `95%` used |

Remote project:

```text
/root/autodl-tmp/nanochat
```

Remote run directory:

```text
/root/autodl-tmp/nanochat_runs
```

## Backup

Backed up the four most useful optimizer states to local Mac:

```text
remote_backups/nanochat_optimizer_states_20260601
```

Backed up files:

| Remote optimizer state | Reason |
|---|---|
| `base_checkpoints/d12_pretrain/optim_002063_rank0.pt` | final D12 pretrain optimizer |
| `chatsft_checkpoints/d12_pretrain/optim_001000_rank0.pt` | general SFT optimizer |
| `chatsft_checkpoints/d12_math_tool_warmup/optim_000033_rank0.pt` | pure calculator warmup optimizer |
| `chatsft_checkpoints/d12_math_tool_mixed/optim_000105_rank0.pt` | mixed calculator warmup optimizer |

Local backup size:

```text
3.4G
```

## Remote Deletion

Deleted all remote optimizer states:

```text
find /root/autodl-tmp/nanochat_runs -name 'optim_*.pt' -print -delete
```

Deleted remote files:

```text
/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_gpu_smoke/optim_000002_rank0.pt
/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_gpu_smoke_b8/optim_000002_rank0.pt
/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_pretrain/optim_000500_rank0.pt
/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_pretrain/optim_001000_rank0.pt
/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_pretrain/optim_001500_rank0.pt
/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_pretrain/optim_002000_rank0.pt
/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_pretrain/optim_002063_rank0.pt
/root/autodl-tmp/nanochat_runs/chatsft_checkpoints/d12_pretrain/optim_001000_rank0.pt
/root/autodl-tmp/nanochat_runs/chatsft_checkpoints/d12_math_tool_warmup/optim_000033_rank0.pt
/root/autodl-tmp/nanochat_runs/chatsft_checkpoints/d12_math_tool_mixed/optim_000105_rank0.pt
```

## Result

Remote data disk after cleanup:

| Metric | Before | After |
|---|---:|---:|
| Used | `48G` | `39G` |
| Available | `2.9G` | `12G` |
| Use% | `95%` | `78%` |

Remaining checkpoint directory sizes:

| Directory | Size |
|---|---:|
| `base_checkpoints/d12_gpu_smoke` | `586M` |
| `base_checkpoints/d12_gpu_smoke_b8` | `586M` |
| `chatsft_checkpoints/d12_math_tool_mixed` | `586M` |
| `chatsft_checkpoints/d12_math_tool_warmup` | `586M` |
| `chatsft_checkpoints/d12_pretrain` | `586M` |
| `base_checkpoints/d12_pretrain` | `2.9G` |

No remote `optim_*.pt` files remain.

## Notes

- All remote `model_*.pt` and `meta_*.json` files were preserved.
- These checkpoints can still be used for inference/evaluation and for continued training from weights.
- Without remote optimizer states, exact optimizer-momentum resume is not available on the server unless the backed-up optimizer files are restored from local Mac.

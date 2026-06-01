# d12 Pretrain Summary

Date: 2026-05-30  
Remote project: `/root/autodl-tmp/nanochat`  
Run name: `d12_pretrain`  
Final checkpoint: `/root/autodl-tmp/nanochat_runs/base_checkpoints/d12_pretrain/model_002063.pt`  
Remote log: `/root/autodl-tmp/nanochat/logs/d12_pretrain_live.log`  
W&B offline run: `/root/autodl-tmp/nanochat/wandb/offline-run-20260530_165803-kix3rf6a`

## 1. Run Setup

This run trained a d12 decoder-only Transformer base model on a single NVIDIA GeForce RTX 4090 D 24GB GPU.

Model configuration:

| Item | Value |
|---|---:|
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 6 |
| KV heads | 3 |
| Head dim | 128 |
| Sequence length | 2048 |
| Vocabulary size | 32768 |
| Window pattern | `SSSL` |

Parameter counts from the training log:

| Parameter group | Count |
|---|---:|
| Token embedding `wte` | 25,165,824 |
| Value embeddings | 75,497,472 |
| LM head | 25,165,824 |
| Transformer matrices | 77,857,344 |
| Scalars | 24 |
| Total params | 203,686,488 |
| Scaling params | 103,023,168 |

`scaling_params` is the parameter subset used for the token budget heuristic: `transformer_matrices + lm_head = 77,857,344 + 25,165,824 = 103,023,168`.

Training configuration:

| Item | Value |
|---|---:|
| Per-device batch size | 8 |
| Tokens per micro-batch | 16,384 |
| Total batch size | 524,288 tokens |
| Gradient accumulation | 32 |
| Training steps | 2,063 |
| Training tokens | 1,081,606,144 |
| Token / scaling-param ratio | 10.50 |
| Estimated training FLOPs | `8.216927e17` |
| Final epoch counter | 3 |
| Optimizer setup | Muon + AdamW mixed optimizer |
| LR schedule | warmdown to zero over final 50% |

Important runtime note: FlashAttention3 was not available on this server, so the run used PyTorch SDPA fallback. The log also warned that SDPA does not support the configured sliding-window pattern efficiently. This means the run is valid as a pretraining correctness/stability result, but not a proof of the best possible sliding-window training throughput.

## 2. Main Results

The pretraining run completed successfully.

| Metric | Result |
|---|---:|
| Final validation BPB | 0.937980 |
| Minimum validation BPB | 0.937980 |
| Final CORE metric | 0.1116 |
| Final smoothed train loss | 2.91072 |
| Peak GPU memory | 15,220.97 MiB |
| Total training time | 195.46 min |
| Late-run throughput | about 91.7k tokens/sec |
| Late-run BF16 MFU | about 42.17% |

Validation BPB trend:

| Step | Validation BPB |
|---:|---:|
| 0 | 3.184450 |
| 250 | 1.136090 |
| 500 | 1.058270 |
| 750 | 1.025578 |
| 1000 | 1.006899 |
| 1250 | 0.985597 |
| 1500 | 0.965480 |
| 1750 | 0.950073 |
| 2000 | 0.939049 |
| 2063 | 0.937980 |

CORE trend:

| Step | CORE |
|---:|---:|
| 1000 | 0.0994 |
| 2000 | 0.1108 |
| 2063 | 0.1116 |

Final CORE task snapshot:

| Task | Accuracy | Centered |
|---|---:|---:|
| hellaswag_zeroshot | 0.3200 | 0.0933 |
| bigbench_qa_wikidata | 0.2600 | 0.2600 |
| arc_easy | 0.5100 | 0.3467 |
| arc_challenge | 0.2500 | 0.0000 |
| copa | 0.5500 | 0.1000 |
| commonsense_qa | 0.2700 | 0.0875 |
| piqa | 0.5200 | 0.0400 |
| openbook_qa | 0.2900 | 0.0533 |
| lambada_openai | 0.3300 | 0.3300 |
| hellaswag | 0.3100 | 0.0800 |
| boolq | 0.6700 | 0.1316 |
| bigbench_language_identification | 0.2800 | 0.2079 |

## 3. Qualitative Samples

The final checkpoint produced coherent but repetitive base-model continuations.

Examples observed in the log:

- "The capital of France is Paris..." was correct but repeated.
- "The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune" was correct.
- "The opposite of hot is cold..." was initially correct but then self-contradicted through repetition.
- "If 5*x + 3 = 13..." did not solve the equation and fell into repetition.

Interpretation: this is expected for a base pretrain model. It has learned language modeling and some factual continuations, but it has not yet learned instruction-following, robust dialogue formatting, or reliable math reasoning. Those should be evaluated after SFT and tool-use training.

## 4. What This Run Proves

This run provides evidence for the following project claims:

- The end-to-end base pretraining pipeline works: tokenizer artifacts, data loader, model build, optimizer, checkpointing, validation BPB, CORE evaluation, and sample generation all completed.
- The d12 / 203.7M-parameter model fits on a single 24GB GPU with batch size 8, sequence length 2048, and gradient accumulation 32.
- Muon + AdamW training was stable for the whole run; BPB decreased smoothly from 3.184450 to 0.937980.
- The model learned meaningful language modeling behavior: validation BPB kept improving through the final checkpoint, CORE improved from 0.0994 at step 1000 to 0.1116 at the end, and samples became readable.
- The current base model is a good starting point for SFT, ChatCORE evaluation, calculator/tool-use, and inference KV-cache experiments.

## 5. What This Run Does Not Yet Prove

This run should not be overstated.

- It does not prove strong chat ability, because SFT has not been run yet.
- It does not prove strong math ability, because the base model still fails simple algebra-style generation.
- It does not isolate the effect of sliding-window attention, because there is no ablation against full attention yet.
- It does not prove maximum inference speed or memory benefit, because this was a training run and FA3 was unavailable.
- It does not prove broad generalization, because only 11 FineWeb-Edu shards were used and the final epoch counter reached 3, meaning the data was recycled.
- CORE was run with a small per-task cap during training, so it is useful as a trend indicator rather than a final benchmark number.

## 6. Interview Explanation

A concise explanation:

> 我训练了一个 d12 规模的轻量级 decoder-only LLM，约 203.7M total params，其中 scaling params 约 103M。训练预算按 token/scaling-param ratio 10.5 设置，最终训练约 1.08B tokens。在单张 4090D 24GB 上，使用 batch size 8、seq len 2048、梯度累积 32，完整跑完 2063 steps。验证 BPB 从 3.18 降到 0.938，CORE 从中途 0.099 提升到 0.112，说明预训练链路和优化器配置是稳定有效的。当前模型已经具备基础语言续写能力，但还不是 chat model，下一步需要通过 SFT 和 calculator tool-use 验证指令跟随与数学能力。

If asked "这次训练最重要的收获是什么？":

> 最大收获不是模型能力本身，而是我把小模型从 tokenizer、pretrain、checkpoint、BPB/CORE eval 到 sample generation 的闭环跑通了，并且知道每个指标说明什么。BPB 证明语言建模在收敛，CORE 和样例生成用于观察能力趋势，显存和吞吐说明这个规模在 24GB 单卡上是可控的。

If asked "为什么数学还不行？":

> 这是 base pretrain，不是指令模型，也没有经过 tool-use SFT。base model 本质是在做续写，它可以学到局部文本模式和部分事实，但不会稳定遵循问题求解格式。数学能力需要后续在 GSM8K/tool-call 数据上 SFT，并让模型学会在 `<|python_start|>` 和 `<|python_end|>` 之间调用计算器。

## 7. Recommended Next Checks

Before SFT, a clean base eval can be run later if needed:

```bash
cd /root/autodl-tmp/nanochat
source /etc/network_turbo
export NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat_runs
.venv/bin/python -m scripts.base_eval \
  --model-tag d12_pretrain \
  --eval core,bpb,sample \
  --device-batch-size 8 \
  --max-per-task 200 \
  --split-tokens 1048576
```

Recommended next experiment sequence:

1. Run SFT from `d12_pretrain`.
2. Compare pretrain vs SFT on ChatCORE, GSM8K, MMLU, SpellingBee, and dialogue samples.
3. Add calculator/tool-use evaluation for math tasks.
4. Run sliding-window KV-cache inference benchmark and compare peak memory / tokens per second against full-context cache.

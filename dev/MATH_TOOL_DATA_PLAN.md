# Math Tool-Use Data Plan

## Goal

Current SFT checkpoint `d12_pretrain/model_001000.pt` has:

| Metric | Value |
|---|---:|
| Math tool-call rate | `0.00%` |
| Math tool-output rate | `0.00%` |
| Math answer accuracy | `0.00%` |

The immediate bottleneck is not calculator execution. The model does not emit the nanochat tool protocol:

```text
<|python_start|> expression <|python_end|>
```

Therefore the next SFT stage should first train calculator triggering and expression generation.

## Generated Data

Generated files:

| File | Rows | Purpose |
|---|---:|---|
| `data/math_tool/calculator_warmup_train.jsonl` | `30,000` | Train calculator warmup |
| `data/math_tool/calculator_warmup_val.jsonl` | `1,000` | Validate calculator warmup |
| `data/math_tool/calculator_warmup_summary.json` | - | Counts and examples |

Generation command:

```bash
python -m scripts.generate_math_tool_data \
  --output-dir data/math_tool \
  --direct-train 10000 \
  --word-train 20000 \
  --direct-val 500 \
  --word-val 500
```

## Data Mix

Training split:

| Source | Rows | Ratio | Purpose |
|---|---:|---:|---|
| Direct arithmetic | `10,000` | `33.3%` | Teach tool-call triggering on explicit arithmetic |
| Simple word problem | `20,000` | `66.7%` | Teach natural language to expression construction |

The first warmup stage intentionally excludes hard multi-step GSM8K. The objective is to move tool-call rate from `0%` upward before adding harder reasoning distribution.

## Format

The JSONL stores structure-preserving conversations, not plain assistant strings.

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "A family buys 5 adult tickets at $27 each and 5 child tickets at $4 each. What is the total cost? Give the final answer after ####."
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Calculate adult ticket cost plus child ticket cost: "},
        {"type": "python", "text": "5*27+5*4"},
        {"type": "python_output", "text": "155"},
        {"type": "text", "text": "\nThe final answer is:\n\n#### 155"}
      ]
    }
  ],
  "meta": {
    "source": "synthetic_word",
    "kind": "tickets",
    "split": "train",
    "expr": "5*27+5*4",
    "answer": "155"
  }
}
```

When rendered by `tokenizer.render_conversation`, this becomes:

```text
<|assistant_start|>...<|python_start|>5*27+5*4<|python_end|><|output_start|>155<|output_end|>
The final answer is:

#### 155<|assistant_end|>
```

Training mask check:

| Token / span | Supervised? | Why |
|---|---:|---|
| `<|python_start|>` | yes | Model must learn to trigger the tool |
| Python expression | yes | Model must learn expression construction |
| `<|python_end|>` | yes | Model must close the tool call |
| `<|output_start|> result <|output_end|>` | no | Result is forced by engine at inference time |
| Final answer text | yes | Model must integrate tool result into response |

## Loader

Added `tasks.math_tool.MathToolJSON` because the existing `CustomJSON` loader only supports string assistant messages. The math tool data needs assistant content lists with `text`, `python`, and `python_output` parts.

## Next Training Use

Recommended first training stage:

| Component | Ratio |
|---|---:|
| `calculator_warmup_train.jsonl` | `85%-90%` |
| SmolTalk sample / identity | `10%-15%` |

Run only `300-500` optimizer steps first, then immediately evaluate with:

```bash
python -m scripts.tool_use_smoke \
  --source=sft \
  --model-tag=d12_pretrain \
  --step=<new_step> \
  --device-type=cuda \
  --temperature=0.0 \
  --top-k=1
```

Success criterion for moving to GSM8K-heavy SFT:

| Metric | Target |
|---|---:|
| Math tool-call rate | `>60%` |
| Math tool-output rate | `>60%` |
| Non-math tool-call count | close to `0` |

## 2026-05-31 Remote Smoke Results

Remote environment:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D 24GB |
| Project | `/root/autodl-tmp/nanochat` |
| Base SFT checkpoint | `/root/autodl-tmp/nanochat_runs/chatsft_checkpoints/d12_pretrain/model_001000.pt` |

### Pure Calculator Warmup

Command shape:

```bash
python -m scripts.chat_sft \
  --source=sft \
  --model-tag=d12_pretrain \
  --model-step=1000 \
  --output-tag=d12_math_tool_warmup \
  --load-optimizer=0 \
  --math-tool-train=data/math_tool/calculator_warmup_train.jsonl \
  --math-tool-val=data/math_tool/calculator_warmup_val.jsonl \
  --math-tool-epochs=1 \
  --smoltalk-rows=0 \
  --identity-epochs=0 \
  --simple-spelling-size=0 \
  --spellingbee-size=0 \
  --device-batch-size=4 \
  --total-batch-size=65536 \
  --num-iterations=2400 \
  --eval-every=100 \
  --eval-tokens=131072 \
  --chatcore-every=-1 \
  --warmup-ratio=0.03 \
  --warmdown-ratio=0.2
```

Result:

| Metric | Value |
|---|---:|
| Saved checkpoint | `d12_math_tool_warmup/model_000033.pt` |
| Steps | `33` |
| Initial math-tool val BPB | `0.8042` |
| Final/min math-tool val BPB | `0.1662` |
| Peak memory | `8794.46 MiB` |
| Tool smoke math tool-call rate | `100.00%` |
| Tool smoke math answer accuracy | `80.00%` |
| Non-math tool calls | `1/1` |

Interpretation:

- Calculator triggering was learned very quickly.
- The model over-called the tool on non-math prompts, so pure calculator warmup is too narrow.

### Mixed Warmup

Command shape:

```bash
python -m scripts.chat_sft \
  --source=sft \
  --model-tag=d12_math_tool_warmup \
  --model-step=33 \
  --output-tag=d12_math_tool_mixed \
  --load-optimizer=0 \
  --math-tool-train=data/math_tool/calculator_warmup_train.jsonl \
  --math-tool-val=data/math_tool/calculator_warmup_val.jsonl \
  --math-tool-epochs=1 \
  --smoltalk-rows=5000 \
  --identity-epochs=1 \
  --simple-spelling-size=0 \
  --spellingbee-size=0 \
  --device-batch-size=4 \
  --total-batch-size=65536 \
  --num-iterations=3200 \
  --eval-every=100 \
  --eval-tokens=131072 \
  --chatcore-every=-1 \
  --warmup-ratio=0.03 \
  --warmdown-ratio=0.2
```

Result:

| Metric | Value |
|---|---:|
| Saved checkpoint | `d12_math_tool_mixed/model_000105.pt` |
| Steps | `105` |
| Initial math-tool val BPB | `0.1662` |
| Final/min math-tool val BPB | `0.1574 / 0.1568` |
| Peak memory | `8794.46 MiB` |
| Tool smoke math tool-call rate | `80.00%` |
| Tool smoke math answer accuracy | `80.00%` |
| Non-math tool calls | `0/1` |

Interpretation:

- Adding a small amount of general conversation/identity data reduced non-math over-calling.
- The mixed checkpoint correctly solved direct arithmetic and simple one-step word problems with calculator calls.
- It still failed the harder GSM8K-style multi-step prompt, so the next stage should introduce GSM8K-heavy tool-use data.

Local copied logs:

| File | Purpose |
|---|---|
| `dev/d12_math_tool_warmup.log` | pure calculator warmup training log |
| `dev/d12_math_tool_warmup_smoke.jsonl` | pure warmup smoke outputs |
| `dev/d12_math_tool_mixed.log` | mixed warmup training log |
| `dev/d12_math_tool_mixed_smoke.jsonl` | mixed warmup smoke outputs |

Remote disk note:

- `/root/autodl-tmp` was `48G/50G` used after the two warmup checkpoints.
- Each warmup checkpoint directory is about `1.5G`, mostly optimizer state.

## GSM8K-Heavy Stage Prep

Prepared on `2026-06-01` for the next GPU run.

### Bridge Data

Generated GSM8K bridge data:

| File | Rows | Purpose |
|---|---:|---|
| `data/math_tool/gsm8k_bridge_train.jsonl` | `10,000` | Multi-step tool-use training |
| `data/math_tool/gsm8k_bridge_val.jsonl` | `500` | Multi-step tool-use validation |
| `data/math_tool/gsm8k_bridge_summary.json` | - | Counts and examples |

Generation command:

```bash
python -m scripts.generate_gsm8k_bridge_data \
  --output-dir data/math_tool \
  --train-size 10000 \
  --val-size 500
```

Bridge data design:

| Property | Value |
|---|---|
| Tool calls per example | `3` |
| Prompt style | simplified GSM8K-like word problems |
| Answer format | `#### <number>` |
| Main purpose | teach intermediate-result chaining before raw GSM8K-heavy training |

Bridge problem types:

| Kind | Rows |
|---|---:|
| `boxes_sell_buy` | about `1.6k` |
| `buy_then_share` | about `1.7k` |
| `classroom_groups` | about `1.7k` |
| `pages_remaining` | about `1.6k` |
| `seashell_fraction` | about `1.7k` |
| `work_earn_spend` | about `1.7k` |

Example shape:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Ava buys 3 packs of marbles with 12 marbles in each pack. She also gets 28 extra marbles. She shares all the marbles equally among 4 friends. How many marbles does each friend get? Give the final answer after ####."
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "First compute the number from packs: "},
        {"type": "python", "text": "3*12"},
        {"type": "python_output", "text": "36"},
        {"type": "text", "text": " The packs contain 36 marbles.\n"},
        {"type": "text", "text": "Add the extra amount: "},
        {"type": "python", "text": "36+28"},
        {"type": "python_output", "text": "64"},
        {"type": "text", "text": " There are 64 marbles total.\n"},
        {"type": "text", "text": "Divide equally among the friends: "},
        {"type": "python", "text": "64/4"},
        {"type": "python_output", "text": "16"},
        {"type": "text", "text": " Each friend gets 16 marbles."},
        {"type": "text", "text": "\nThe final answer is:\n\n#### 16"}
      ]
    }
  ]
}
```

Validation completed:

| Check | Result |
|---|---|
| `MathToolJSON` loads train/val | pass |
| First 100 examples have 3 python calls | pass |
| First 100 examples have 3 python outputs | pass |
| Rendered text contains `<|python_start|>` | pass |
| Rendered text contains `<|output_start|>` | pass |
| `<|output_start|>` remains unsupervised | pass |

### Proposed GSM8K-Heavy Mixture

Use `d12_math_tool_mixed/model_000105.pt` as the starting checkpoint.

Recommended next command shape:

```bash
python -m scripts.chat_sft \
  --source=sft \
  --model-tag=d12_math_tool_mixed \
  --model-step=105 \
  --output-tag=d12_gsm8k_tool_sft \
  --load-optimizer=0 \
  --math-tool-train=data/math_tool/gsm8k_bridge_train.jsonl,data/math_tool/calculator_warmup_train.jsonl \
  --math-tool-val=data/math_tool/gsm8k_bridge_val.jsonl,data/math_tool/calculator_warmup_val.jsonl \
  --math-tool-epochs=1 \
  --include-gsm8k=1 \
  --gsm8k-epochs=4 \
  --smoltalk-rows=8000 \
  --identity-epochs=1 \
  --simple-spelling-size=0 \
  --spellingbee-size=0 \
  --device-batch-size=4 \
  --total-batch-size=65536 \
  --num-iterations=4000 \
  --eval-every=100 \
  --eval-tokens=131072 \
  --chatcore-every=-1 \
  --warmup-ratio=0.03 \
  --warmdown-ratio=0.2 \
  --run=d12_gsm8k_tool_sft
```

Expected train mixture size:

| Component | Rows |
|---|---:|
| GSM8K bridge | `10,000` |
| Calculator warmup | `30,000` |
| GSM8K train x4 | about `29,000` |
| SmolTalk sample | `8,000` |
| Identity | about `1,000` |
| Total | about `78,000` |

Primary success criteria:

| Metric | Target |
|---|---:|
| Tool smoke math answer accuracy | stay around `80%+` |
| Tool smoke non-math tool calls | `0/1` |
| GSM8K-style smoke prompt | starts producing coherent multi-step tool calls |
| GSM8K subset eval | improves from `0%` baseline |

## 2026-06-01 GSM8K-Heavy Run Result

Remote environment:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D 24GB |
| Start checkpoint | `d12_math_tool_mixed/model_000105.pt` |
| Output checkpoint | `d12_gsm8k_tool_sft/model_000247.pt` |
| Remote model path | `/root/autodl-tmp/nanochat_runs/chatsft_checkpoints/d12_gsm8k_tool_sft/model_000247.pt` |

Training command:

```bash
python -m scripts.chat_sft \
  --source=sft \
  --model-tag=d12_math_tool_mixed \
  --model-step=105 \
  --output-tag=d12_gsm8k_tool_sft \
  --load-optimizer=0 \
  --math-tool-train=data/math_tool/gsm8k_bridge_train.jsonl,data/math_tool/calculator_warmup_train.jsonl \
  --math-tool-val=data/math_tool/gsm8k_bridge_val.jsonl,data/math_tool/calculator_warmup_val.jsonl \
  --math-tool-epochs=1 \
  --include-gsm8k=1 \
  --gsm8k-epochs=4 \
  --smoltalk-rows=8000 \
  --identity-epochs=1 \
  --simple-spelling-size=0 \
  --spellingbee-size=0 \
  --device-batch-size=4 \
  --total-batch-size=65536 \
  --num-iterations=4000 \
  --eval-every=100 \
  --eval-tokens=131072 \
  --chatcore-every=-1 \
  --warmup-ratio=0.03 \
  --warmdown-ratio=0.2 \
  --run=d12_gsm8k_tool_sft
```

Training result:

| Metric | Value |
|---|---:|
| Training mixture | `78,892` rows |
| Steps completed | `247` |
| Initial validation BPB | `0.5644` |
| Step 100 validation BPB | `0.1073` |
| Step 200 validation BPB | `0.1043` |
| Final/min validation BPB | `0.1034` |
| Peak memory | `8794.46 MiB` |
| Throughput | about `87.6k tok/sec` |
| Total training time | `2.95 min` |

Tool-use smoke result:

| Metric | `d12_math_tool_mixed` | `d12_gsm8k_tool_sft` |
|---|---:|---:|
| Math tool-call rate | `80.00%` | `100.00%` |
| Math tool-output rate | `80.00%` | `100.00%` |
| Math answer accuracy | `80.00%` | `80.00%` |
| Non-math tool calls | `0/1` | `0/1` |
| GSM8K-style smoke prompt | failed | correct |

GSM8K official subset:

| Eval | Result |
|---|---:|
| GSM8K test, first 100 examples | `2/100 = 2.00%` |

20-example diagnostic sample:

| Metric | Value |
|---|---:|
| Correct | `1/20` |
| Examples with tool calls | `18/20` |
| Average python calls/example | `3.45` |
| Examples with no parsed final answer | `2/20` |

Diagnosis:

- The run achieved the main protocol goal: the model now reliably emits nanochat calculator tool tokens and can solve the curated GSM8K-style smoke prompt with three coherent tool calls.
- Official GSM8K improved from the prior `0%` baseline, but only to `2%` on the first 100 examples.
- Error samples show the dominant failure is not calculator execution. The model often calls the tool, but reads the problem incorrectly, chooses the wrong quantities, or constructs the wrong expression.
- Some failures contain excessive repeated tool calls or non-integer/format drift, especially on harder fraction, ratio, time, and multi-entity problems.

Local copied logs:

| File | Purpose |
|---|---|
| `dev/d12_gsm8k_tool_sft.log` | training log |
| `dev/d12_gsm8k_tool_sft_smoke.jsonl` | smoke outputs |
| `dev/d12_gsm8k_tool_sft_gsm8k_eval100.log` | GSM8K first-100 eval |
| `dev/d12_gsm8k_tool_sft_gsm8k_samples20.jsonl` | diagnostic generations |

Remote cleanup:

- The run created `optim_000247_rank0.pt` under `d12_gsm8k_tool_sft`.
- That optimizer state was deleted after training to preserve remote disk space.
- `model_000247.pt` and `meta_000247.json` were preserved.
- Remote data disk after cleanup was about `40G/50G`, `11G` available.

Next experiment direction:

- Add harder synthetic bridge categories matching observed GSM8K failures: ratios, fractions, percentages, time intervals, and multi-entity bookkeeping.
- Add negative/format examples to reduce excessive repeated tool calls.
- Evaluate expression accuracy separately from final-answer accuracy so failures can be attributed to problem parsing vs arithmetic execution vs final formatting.

## 2026-06-01 Hard Bridge GSM8K Run Result

This run was designed after the first GSM8K-heavy checkpoint showed that the model had learned the calculator protocol but still failed on problem understanding. The hard bridge data focuses on ratio, fraction, percentage, time interval, multi-entity, and two-item cost patterns.

Generated hard bridge files:

| File | Rows | Purpose |
|---|---:|---|
| `data/math_tool/gsm8k_hard_bridge_train.jsonl` | `30,000` | Harder multi-step tool-use training |
| `data/math_tool/gsm8k_hard_bridge_val.jsonl` | `1,000` | Harder multi-step validation |
| `data/math_tool/gsm8k_hard_bridge_summary.json` | - | Counts and examples |

Generation command:

```bash
python -m scripts.generate_gsm8k_hard_bridge_data \
  --output-dir data/math_tool \
  --train-size 30000 \
  --val-size 1000
```

Training command:

```bash
python -m scripts.chat_sft \
  --source=sft \
  --model-tag=d12_gsm8k_tool_sft \
  --model-step=247 \
  --output-tag=d12_gsm8k_hard_tool_sft \
  --load-optimizer=0 \
  --math-tool-train=data/math_tool/gsm8k_hard_bridge_train.jsonl,data/math_tool/gsm8k_bridge_train.jsonl,data/math_tool/calculator_warmup_train.jsonl \
  --math-tool-val=data/math_tool/gsm8k_hard_bridge_val.jsonl,data/math_tool/gsm8k_bridge_val.jsonl,data/math_tool/calculator_warmup_val.jsonl \
  --math-tool-epochs=1 \
  --include-gsm8k=1 \
  --gsm8k-epochs=2 \
  --smoltalk-rows=10000 \
  --identity-epochs=1 \
  --simple-spelling-size=0 \
  --spellingbee-size=0 \
  --device-batch-size=4 \
  --total-batch-size=65536 \
  --num-iterations=5000 \
  --eval-every=100 \
  --eval-tokens=131072 \
  --chatcore-every=-1 \
  --warmup-ratio=0.03 \
  --warmdown-ratio=0.2 \
  --run=d12_gsm8k_hard_tool_sft
```

Training result:

| Metric | Value |
|---|---:|
| Training mixture | `95,946` rows |
| Start checkpoint | `d12_gsm8k_tool_sft/model_000247.pt` |
| Output checkpoint | `d12_gsm8k_hard_tool_sft/model_000293.pt` |
| Remote model path | `/root/autodl-tmp/nanochat_runs/chatsft_checkpoints/d12_gsm8k_hard_tool_sft/model_000293.pt` |
| Steps completed | `293` |
| Initial validation BPB | `0.5482` |
| Step 200 validation BPB | `0.0789` |
| Final/min validation BPB | `0.0778` |
| Peak memory | `8794.46 MiB` |
| Throughput | about `87.2k tok/sec` |
| Total training time | `3.54 min` |

Evaluation:

| Metric | Previous GSM8K-heavy | Hard bridge run |
|---|---:|---:|
| Tool smoke math tool-call rate | `100.00%` | `100.00%` |
| Tool smoke math answer accuracy | `80.00%` | `80.00%` |
| Tool smoke non-math tool calls | `0/1` | `0/1` |
| GSM8K test, first 100 examples | `2/100 = 2.00%` | `4/100 = 4.00%` |
| 20-example diagnostic correct | `1/20` | `2/20` |
| 20-example examples with tool calls | `18/20` | `20/20` |
| 20-example average python calls/example | `3.45` | `3.45` |

Diagnosis:

- The hard bridge run improved GSM8K first-100 accuracy from `2%` to `4%`, and the diagnostic sample from `1/20` to `2/20`.
- Tool-use behavior is now very stable: the model calls the calculator on all sampled math problems and avoids calling it on the non-math smoke prompt.
- The remaining bottleneck is problem-to-expression translation. Typical failures choose the wrong quantities, repeat operations, ignore unit conversion details, or stop after an intermediate value.
- The model often executes calculator calls correctly once the expression is formed, so simply adding more calculator-format data is unlikely to be enough. The next useful data should target semantic decomposition and expression planning.

Local copied logs:

| File | Purpose |
|---|---|
| `dev/d12_gsm8k_hard_tool_sft.log` | training log |
| `dev/d12_gsm8k_hard_tool_sft_smoke.jsonl` | smoke outputs |
| `dev/d12_gsm8k_hard_tool_sft_gsm8k_eval100.log` | GSM8K first-100 eval |
| `dev/d12_gsm8k_hard_tool_sft_gsm8k_samples20.jsonl` | diagnostic generations |

Remote cleanup:

- The run created `optim_000293_rank0.pt` under `d12_gsm8k_hard_tool_sft`.
- That optimizer state was deleted after copying logs to preserve remote disk space.
- `model_000293.pt` and `meta_000293.json` were preserved.
- Remote data disk after cleanup was about `40G/50G`, `11G` available.

Next experiment direction:

- Build a smaller but higher-quality decomposition dataset where each step explicitly names the variable and checks the unit before tool use.
- Add official GSM8K-style SFT examples with tool traces produced by a stronger teacher, not only synthetic templates.
- Evaluate by category, especially ratio, percentage, time, and multi-entity bookkeeping, instead of relying only on aggregate GSM8K accuracy.

## 2026-06-01 DeepSeek Trace Pilot

中文详细记录见：

```text
dev/GSM8K_DEEPSEEK_TRACE_DATA_20260601.md
```

本次新增 `scripts/generate_gsm8k_deepseek_traces.py`，用 DeepSeek 将官方 GSM8K train 样本重写为 decomposition tool trace。脚本只采信 DeepSeek 生成的步骤解释和表达式，`python_output` 由本地安全表达式求值生成，并拒收最终答案不匹配或结构非法的样本。

Pilot 结果：

| Split | 成功样本 | 拒收样本 | 说明 |
|---|---:|---:|---|
| train | `86` | - | `gsm8k_deepseek_trace_pilot_train.jsonl` |
| val | `17` | - | `gsm8k_deepseek_trace_pilot_val.jsonl` |
| rejects | - | `17` | API 空返回、不完整 JSON 等 |

结构校验已通过：`MathToolJSON` 能加载全部成功样本，且所有样本的 `python` 与 `python_output` 数量一致。

### v1 本地数据

继续本地扩展 DeepSeek trace 数据时，DeepSeek API 后半段返回 `402 Payment Required`，因此本轮保留已成功生成的样本，并整理出可用于下一轮 SFT 的本地 split：

| 文件 | Rows |
|---|---:|
| `data/math_tool/gsm8k_deepseek_trace_v1_sft_train.jsonl` | `367` |
| `data/math_tool/gsm8k_deepseek_trace_v1_sft_val.jsonl` | `50` |
| `data/math_tool/gsm8k_deepseek_trace_v1_sft_summary.json` | - |

结构校验通过，坏样本为 `0`。这批数据可以作为下一轮小规模 decomposition SFT 的主训练信号。

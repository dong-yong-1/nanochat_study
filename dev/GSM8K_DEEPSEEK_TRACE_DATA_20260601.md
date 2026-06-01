# GSM8K DeepSeek Tool Trace 数据生成记录

日期：2026-06-01

## 目标

这一步不是继续简单复用作者内置的 GSM8K 数据，而是把官方 GSM8K train 样本重写成更适合小模型学习的 decomposition tool trace 数据。

之前实验已经说明：

- 模型已经学会稳定调用 calculator tool；
- 但 GSM8K 真实题目上仍然经常拿错变量、理解错关系或构造错表达式；
- 所以下一步需要强化“题意解析 -> 变量定义 -> 表达式规划”，而不是只继续堆原始 GSM8K。

## 生成脚本

新增脚本：

```text
scripts/generate_gsm8k_deepseek_traces.py
```

脚本逻辑：

1. 从官方 GSM8K train 抽样；
2. 调用 DeepSeek API 生成 step decomposition；
3. DeepSeek 只负责生成解释和表达式；
4. 每个 `python_output` 都由脚本本地安全计算；
5. 如果最终答案和官方 GSM8K `#### answer` 不一致，则拒收；
6. 输出 nanochat 可直接训练的 JSONL 结构。

安全约束：

- 表达式只允许数字、`+ - * /` 和括号；
- 不允许变量、函数、赋值或任意 Python 代码；
- 每条样本限制 2 到 6 个工具调用；
- 拒收 JSON 格式错误、表达式非法、答案不一致的样本。

## API 配置

脚本从 `.env` 读取：

```text
DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL
DEEPSEEK_MODEL
```

本次使用：

```text
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

## Pilot 生成结果

命令形态：

```bash
.venv/bin/python -m scripts.generate_gsm8k_deepseek_traces \
  --name-prefix gsm8k_deepseek_trace_pilot \
  --train-size 100 \
  --val-size 20 \
  --limit=120 \
  --workers=6 \
  --arrow-path /Users/dongyong/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow \
  --resume
```

生成文件：

| 文件 | 数量 | 说明 |
|---|---:|---|
| `data/math_tool/gsm8k_deepseek_trace_pilot_train.jsonl` | `86` | 训练样本 |
| `data/math_tool/gsm8k_deepseek_trace_pilot_val.jsonl` | `17` | 验证样本 |
| `data/math_tool/gsm8k_deepseek_trace_pilot_rejects.jsonl` | `17` | 被拒样本 |
| `data/math_tool/gsm8k_deepseek_trace_pilot_summary.json` | - | 统计摘要 |

拒收原因主要是 API 偶发返回空内容或不完整 JSON，例如 `Expecting value`、`Unterminated string`。这些样本没有进入训练集。

## 结构校验

使用 `MathToolJSON` 加载通过。

| Split | Rows | 2 calls | 3 calls | 4 calls | 5 calls | 6 calls | 坏样本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | `86` | `27` | `35` | `17` | `3` | `4` | `0` |
| val | `17` | `3` | `8` | `4` | `2` | `0` | `0` |

每条样本均满足：

- assistant content 是结构化 list；
- `python` 和 `python_output` 数量一致；
- 至少包含一个工具调用；
- 最终答案格式为 `#### answer`；
- `python_output` 由本地表达式计算得到。

## 样本特点

相比之前的 synthetic bridge，这批数据更接近真实 GSM8K：

- 问题语言来自官方 GSM8K；
- 每一步解释会显式说明当前变量和单位；
- 表达式是从题意关系推导出来，而不是模板填空；
- 多数样本有 2 到 4 个工具调用，更适合训练“必要计算”而不是无脑频繁调用工具。

示例模式：

```text
Calculate Alice's commission from $2500 sales at 2%.
<python>2500 * 2 / 100</python>
<output>50</output>

Compute her total earnings by adding base salary $240 and commission.
<python>240 + 50</python>
<output>290</output>

Determine her savings, which is 10% of total earnings.
<python>290 * 10 / 100</python>
<output>29</output>

#### 29
```

## 当前限制

这只是 pilot 数据，还不够训练出明显变化：

- 规模只有 `86/17`，适合 smoke SFT 或格式验证；
- 因 API/额度限制，本轮未继续补齐到 `100/20`；
- 拒收率约 `17 / 120 = 14.2%`，后续扩数据前应优化 prompt 或 retry 策略；
- 还没有做人工抽样完整审查，训练前建议至少检查 20 条。

## 下一步建议

短期：

1. 人工抽查 20 条 pilot 数据，确认推理质量；
2. 补跑到至少 `500 train / 100 val`；
3. 用 `d12_gsm8k_hard_tool_sft/model_000293.pt` 做一轮小规模 SFT；
4. 对比 GSM8K@100、tool-use smoke 和 20 题诊断。

训练混合建议：

| 数据 | 建议比例 |
|---|---:|
| DeepSeek GSM8K trace | `50%-60%` |
| hard bridge | `15%-20%` |
| 原始 GSM8K | `10%-15%` |
| calculator warmup | `5%-10%` |
| SmolTalk / identity | `10%` |

判断标准：

- 如果 GSM8K@100 从 `4%` 继续上升，说明 decomposition trace 有效；
- 如果 tool-use smoke 保持稳定且非数学不乱调工具，说明没有破坏通用行为；
- 如果 20 题诊断里“拿错变量/关系方向”的错误减少，就可以把这条路线写进简历故事。

## 2026-06-01 v1 本地生成进展

用户确认“数据先在本地生成就可以”后，继续在本地扩展 DeepSeek trace 数据。

本轮目标原本是生成 `500 train / 100 val`，但 DeepSeek API 后半段返回 `402 Payment Required`，说明当前 API key 余额或额度不足。因此本轮停止继续扩展，只保留已经成功通过校验的数据。

命令形态：

```bash
.venv/bin/python -m scripts.generate_gsm8k_deepseek_traces \
  --name-prefix gsm8k_deepseek_trace_v1 \
  --train-size 500 \
  --val-size 100 \
  --limit=600 \
  --workers=8 \
  --arrow-path /Users/dongyong/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow \
  --resume
```

原始生成结果：

| 文件 | 数量 | 说明 |
|---|---:|---|
| `data/math_tool/gsm8k_deepseek_trace_v1_train.jsonl` | `417` | 成功生成的 trace 样本 |
| `data/math_tool/gsm8k_deepseek_trace_v1_rejects.jsonl` | `183` | 被拒样本 |

由于 API 在进入 val 阶段前后余额不足，本轮没有成功生成独立 val 文件。为了能进行下一步 SFT 验证，已从 417 条成功样本中切出本地训练/验证版本：

| 文件 | 数量 | 说明 |
|---|---:|---|
| `data/math_tool/gsm8k_deepseek_trace_v1_sft_train.jsonl` | `367` | SFT 训练样本 |
| `data/math_tool/gsm8k_deepseek_trace_v1_sft_val.jsonl` | `50` | SFT 验证样本 |
| `data/math_tool/gsm8k_deepseek_trace_v1_sft_summary.json` | - | 统计摘要 |

结构校验：

| Split | Rows | 2 calls | 3 calls | 4 calls | 5 calls | 6 calls | 坏样本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | `367` | `125` | `125` | `73` | `32` | `12` | `0` |
| val | `50` | `11` | `17` | `15` | `4` | `3` | `0` |

reject 原因统计：

| 原因 | 数量 |
|---|---:|
| API 余额/额度不足 `402 Payment Required` | `138` |
| 空返回或非 JSON | `36` |
| JSON 格式缺逗号等错误 | `6` |
| JSON 字符串截断 | `3` |

当前判断：

- `367/50` 已经够做一轮小规模 SFT 验证，但还不够作为最终强结论；
- 这批数据比 pilot 更适合验证“高质量 decomposition trace 是否改善题意解析”；
- 如果想继续扩到 `500/100` 或 `1k/100`，需要先给 DeepSeek API 充值或换可用 key；
- 在服务器恢复前，本地已完成下一步训练所需的数据准备。

建议下一步训练起点：

```text
d12_gsm8k_hard_tool_sft/model_000293.pt
```

建议数据混合：

| 数据 | 文件 | 用途 |
|---|---|---|
| DeepSeek decomposition trace | `gsm8k_deepseek_trace_v1_sft_train.jsonl` | 主训练信号 |
| hard bridge | `gsm8k_hard_bridge_train.jsonl` | 保持模板覆盖和稳定工具调用 |
| calculator warmup | `calculator_warmup_train.jsonl` | 保持基础工具格式 |
| SmolTalk/identity | 通过 `chat_sft.py` 参数采样 | 防止非数学过拟合 |

第一轮建议不要训练太久，目标是看趋势：

- tool-use smoke 不能掉；
- GSM8K@100 是否高于 `4%`；
- 20 题诊断中“变量拿错/关系方向错”是否减少。

## 2026-06-01 服务器小规模 SFT 验证

用户开卡后，在服务器上用 `d12_gsm8k_hard_tool_sft/model_000293.pt` 作为起点，验证 v1 decomposition trace 是否能继续提升数学能力。

服务器状态：

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D 24GB |
| 起点 checkpoint | `d12_gsm8k_hard_tool_sft/model_000293.pt` |
| 数据目录 | `/root/autodl-tmp/nanochat/data/math_tool` |

### 训练混合

由于 `367/50` 条 DeepSeek trace 规模较小，如果直接和完整 hard bridge / calculator 数据混合会被淹没，因此远端临时构造了一个小混合数据集：

| 数据 | Rows | 说明 |
|---|---:|---|
| DeepSeek v1 trace repeated | `2,936` | `367 * 8`，主训练信号 |
| hard bridge subset | `1,000` | 保持多步工具调用稳定性 |
| calculator subset | `500` | 保持基础 calculator 格式 |
| train total | `4,436` | math-tool 训练数据 |
| val total | `350` | `50` DeepSeek + `200` hard bridge + `100` calculator |

加上 `SmolTalk 2000` 和 identity 后，训练日志中总 mixture 为 `7,436` rows。

### 实验 A：正常跑完整 mixture

输出 checkpoint：

```text
d12_deepseek_trace_sft/model_000091.pt
```

训练结果：

| 指标 | 值 |
|---|---:|
| Steps | `91` |
| 初始 val BPB | `0.2866` |
| 最低 val BPB | `0.1604` at step `25` |
| 最终 val BPB | `0.2025` |
| 峰值显存 | `8794 MiB` |
| 训练时间 | `0.52 min` |
| GSM8K@100 | `2/100 = 2.00%` |
| smoke 数学准确率 | `80.00%` |
| smoke tool-call rate | `100.00%` |
| 非数学 tool call | `0/1` |

观察：

- step 25 后 val BPB 开始变差，说明小数据重复训练存在过拟合；
- GSM8K@100 从上一轮 hard bridge 的 `4%` 掉到 `2%`；
- smoke 中有一题输出 `#### 16.0`，数值上对但 exact string 评测判错，说明还需要答案归一化或整数格式约束。

### 实验 B：短训版本

为了验证过拟合判断，又跑了一个短训版本：

```text
d12_deepseek_trace_sft_short/model_000006.pt
```

由于数据集实际 token 量较小，这个设置只跑了 `6` step 即完整过一遍数据。

结果：

| 指标 | 值 |
|---|---:|
| Steps | `6` |
| 初始 val BPB | `0.2866` |
| 最终/min val BPB | `0.1792` |
| GSM8K@100 | `0/100 = 0.00%` |
| smoke 数学准确率 | `40.00%` |
| smoke tool-call rate | `80.00%` |
| 非数学 tool call | `0/1` |

观察：

- 短训版本反而破坏了已有 calculator 行为；
- 说明不是“少训一点”就能解决，而是数据规模、配比和学习率都需要重新设计。

### 本轮结论

这轮实验没有提升 GSM8K，反而验证了一个重要负结果：

> 仅有 `367` 条 DeepSeek decomposition trace，并通过重复放大来训练，会破坏已有 tool-use 行为，不能证明 decomposition trace 路线有效。

更准确的判断是：

- DeepSeek trace 数据格式和质量是可用的；
- 但规模太小，重复比例过高；
- 训练配比太激进，DeepSeek trace 主导后模型开始模仿解释风格，但丢失了一部分原有 calculator 稳定性；
- 需要至少扩到 `1k-3k` 条真实 GSM8K trace，或者降低学习率/冻结部分参数/减少重复倍数，并保留更多原始 GSM8K + hard bridge + calculator 数据。

### 下一步建议

下一轮不要继续用 `367` 条数据重复硬训。更合理的方向：

1. 先给 DeepSeek API 充值或换可用 key，把 trace 扩到至少 `1,000 train / 100 val`；
2. 重新设计混合比例，让 DeepSeek trace 只占 `20%-30%`，不要占主导；
3. 降低学习率或减少训练步数，避免破坏已有 tool-use；
4. 修改评测中的答案归一化，把 `16` 和 `16.0` 这类等价答案视作相同，用于辅助分析；
5. 再做 GSM8K@100 与 20 题诊断。

本轮负结果对简历故事仍然有价值：它说明我们不是盲目堆数据，而是通过实验发现“高质量 trace 的规模与配比”是关键因素。

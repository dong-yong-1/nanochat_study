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

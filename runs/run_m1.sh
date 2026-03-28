export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_API_KEY=""
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# ======================== 你只需要运行这一段 ========================
# 下载 8 片数据（完全适配你的 118M 模型）
python -m nanochat.dataset -n 8

# 训练分词器（用 20 亿字符，标准）
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# ======================== 这是为你定制的训练命令 ========================
# 你的模型：118M | depth=12 | embd=512 | heads=8 | seqlen=1024 | SSSL窗口
python -m scripts.base_train \
    --depth=12 \
    --head-dim=64 \
    --aspect-ratio=42 \
    --window-pattern=SSSL \
    --max-seq-len=1024 \
    --device-batch-size=64 \
    --target-param-data-ratio=3.6 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=200 \
    --run=$WANDB_RUN

# 评估脚本
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

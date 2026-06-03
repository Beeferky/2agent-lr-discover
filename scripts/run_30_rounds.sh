#!/usr/bin/env bash
# 连跑 N 轮 agent（默认 30）。每轮间 clean_run 重置训练状态+ckpt+wandb id，
# 但保留 decisions_log / rollback_log / final_results_log 三个 jsonl —— 这是跨 run 记忆，
# 让方案 1 的 _build_cross_run_section 越跑越有历史可参考。
# 用法: bash scripts/run_30_rounds.sh [rounds]
set -u
cd "$(dirname "$0")/.."

# 记住外部显式传入的 DEVICE/MASTER_PORT（必须在 source .env 之前，否则被 .env 覆盖）
_DEVICE_OVERRIDE="${DEVICE:-}"
_PORT_OVERRIDE="${MASTER_PORT:-}"

# cola 环境（关键：torchrun/python 用对 env）
export PATH="/home/bulou_tmp/.conda/envs/cola/bin:$PATH"
# 密钥
set -a && source .env && set +a
# 外部传入 > .env > 默认；并把全角逗号归一化为半角。不设 AGENT_MODE = cosine+bf16+adamw
export DEVICE="${_DEVICE_OVERRIDE:-${DEVICE:-4,5,6,7}}"
export DEVICE="${DEVICE//，/,}"
export MASTER_PORT="${_PORT_OVERRIDE:-${MASTER_PORT:-29500}}"
unset AGENT_MODE

# ── 模型规模 / 经验回放隔离（默认 60M，与原行为一致）──
# 用法: MODEL_CONFIG=cola_configs/llama_20m.json RUN_TAG=llama20m bash scripts/run_30_rounds.sh 30
export MODEL_CONFIG="${MODEL_CONFIG:-cola_configs/llama_60m.json}"
export RUN_TAG="${RUN_TAG:-}"
SFX=""; [ -n "$RUN_TAG" ] && SFX="_${RUN_TAG}"
FINAL_LOG="final_results_log${SFX}.jsonl"

# ── 安全检查：不允许和已有训练并发（两个 agent run 会踩同一组全局文件 + 抢显存）──
# 只匹配我们自己 (python agent.py / -u main.py --model_type llama / torchrun --standalone)，
# 避免误中别人的 ray/dashboard/agent.py 这种带路径的同名文件
EXISTING=$(pgrep -af "python[0-9.]*[[:space:]]+agent\.py|-u[[:space:]]+main\.py --model_type llama|torchrun --standalone --nproc" 2>/dev/null | grep -v "$$" || true)
if [ -n "$EXISTING" ]; then
    echo "[campaign] ✗ 已有训练进程在跑，拒绝启动（避免并发踩 results/agent_run + 抢 GPU）："
    echo "$EXISTING"
    echo "[campaign]   先处理掉它，或确认无误后再跑。"
    exit 1
fi
for g in $(echo "$DEVICE" | tr ',' ' '); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -gt 2000 ]; then
        echo "[campaign] ✗ GPU $g 已用 ${used}MiB（>2000），疑似被占，拒绝启动。"
        echo "[campaign]   nvidia-smi 确认 DEVICE=$DEVICE 的卡空闲后再跑。"
        exit 1
    fi
done

ROUNDS=${1:-30}
CAMPAIGN_TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="session_logs/campaign_${CAMPAIGN_TS}"
mkdir -p "$LOGDIR"

echo "[campaign] start $ROUNDS rounds | model=$MODEL_CONFIG | tag=${RUN_TAG:-<none>} | DEVICE=$DEVICE | logdir=$LOGDIR"

for i in $(seq 1 "$ROUNDS"); do
    TS=$(date +%Y%m%d_%H%M%S)
    RLOG="${LOGDIR}/round_$(printf '%02d' "$i")_${TS}.log"
    echo "==================== ROUND $i / $ROUNDS  ($TS) ===================="
    bash scripts/clean_run.sh
    python agent.py > "$RLOG" 2>&1
    rc=$?
    # 抓本轮 final ppl（agent 会写进 $FINAL_LOG 最后一行）
    ppl=$(tail -n 1 "$FINAL_LOG" 2>/dev/null | grep -oE '"final_eval_ppl":[ ]*[0-9.]+' | grep -oE '[0-9.]+$')
    echo "[campaign] round $i done | exit=$rc | final_ppl=${ppl:-NA} | log=$RLOG"
done

echo "[campaign] ALL $ROUNDS rounds complete. $FINAL_LOG 汇总："
cat "$FINAL_LOG" 2>/dev/null | grep -oE '"final_eval_ppl":[ ]*[0-9.]+' || true

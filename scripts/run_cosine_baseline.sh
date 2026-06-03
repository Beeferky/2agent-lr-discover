#!/usr/bin/env bash
# Cosine decay baseline SWEEP — 顺序跑多个 peak LR，作为 2-agent 实验的对照组。
# 每个 LR 一个独立 wandb run（同 wandb project 内）+ 独立 save_dir。
#
# 用法（全 env 可覆盖）：
#   DEVICE="4,5,6,7" \
#   MODEL_CONFIG=cola_configs/llama_60m.json \
#   LRS="1e-3 3e-3 5e-3 1e-2" \
#   bash scripts/run_cosine_baseline.sh
#
# 单跑某个 LR：LRS="3e-3" bash scripts/run_cosine_baseline.sh
set -u
cd "$(dirname "$0")/.."

# 外部显式传入的 DEVICE/MASTER_PORT 优先（要在 source .env 之前抓）
_DEVICE_OVERRIDE="${DEVICE:-}"
_PORT_OVERRIDE="${MASTER_PORT:-}"

export PATH="/home/bulou_tmp/.conda/envs/cola/bin:$PATH"
set -a && source .env && set +a

export DEVICE="${_DEVICE_OVERRIDE:-${DEVICE:-4,5,6,7}}"
export DEVICE="${DEVICE//，/,}"
export MASTER_PORT="${_PORT_OVERRIDE:-${MASTER_PORT:-29500}}"

# 共享配置
MODEL_CONFIG="${MODEL_CONFIG:-cola_configs/llama_60m.json}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"   # 衰减终点 = 0.1×peak（cosine10x 风格）
WARMUP="${WARMUP:-1000}"
STEPS="${STEPS:-10000}"
WANDB_PROJECT="${WANDB_PROJECT:-2AgentPlan}"

# Sweep 的 peak LR 列表（空格分隔）
LRS="${LRS:-1e-3 3e-3 5e-3 1e-2}"

# 从 MODEL_CONFIG 抽个简短 tag（用于 save_dir/run_name 命名）
MODEL_TAG=$(basename "$MODEL_CONFIG" .json | sed 's/^llama_//')

# 一次性安全检查（sweep 内 LR 切换不重复扫描）
EXISTING=$(pgrep -af "python[0-9.]*[[:space:]]+agent\.py|-u[[:space:]]+main\.py --model_type llama|torchrun --standalone --nproc" 2>/dev/null | grep -v "$$" || true)
if [ -n "$EXISTING" ]; then
    echo "[sweep] ✗ 已有训练进程在跑，拒绝启动："
    echo "$EXISTING"
    exit 1
fi
for g in $(echo "$DEVICE" | tr ',' ' '); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -gt 2000 ]; then
        echo "[sweep] ✗ GPU $g 已用 ${used}MiB，拒绝启动。"
        exit 1
    fi
done

NPROC=$(echo "$DEVICE" | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES=$DEVICE

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="session_logs/cosine_sweep_${SWEEP_TS}"
mkdir -p "$LOGDIR"

echo "==========================================================="
echo " Cosine decay baseline SWEEP (no agent)"
echo "   model      : $MODEL_CONFIG   (tag=$MODEL_TAG)"
echo "   device     : $DEVICE  ($NPROC 卡)"
echo "   peak LRs   : $LRS"
echo "   warmup     : $WARMUP   total steps: $STEPS   min_lr_ratio: $MIN_LR_RATIO"
echo "   wandb      : $WANDB_PROJECT"
echo "   sweep log  : $LOGDIR"
echo "==========================================================="

declare -a RESULTS_TAGS=()
declare -a RESULTS_PPLS=()

for LR in $LRS; do
    RUN_TAG="cosine_lr${LR}_${MODEL_TAG}"
    SAVE_DIR="results/${RUN_TAG}"
    RLOG="${LOGDIR}/${RUN_TAG}_${SWEEP_TS}.log"
    MIN_LR=$(awk "BEGIN{printf \"%.2e\", $LR * $MIN_LR_RATIO}")

    echo ""
    echo "─── [SWEEP] LR=$LR  →  min LR=$MIN_LR ───────────────────"

    if [ -d "$SAVE_DIR" ]; then
        echo "[sweep] ⚠ $SAVE_DIR 已存在，跳过（删除后重跑：rm -rf $SAVE_DIR）"
        continue
    fi

    torchrun --standalone --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" \
        main.py \
        --no_agent \
        --model_type llama \
        --model_config "$MODEL_CONFIG" \
        --scheduler cosine \
        --lr "$LR" \
        --min_lr_ratio "$MIN_LR_RATIO" \
        --warmup_steps "$WARMUP" \
        --num_training_steps "$STEPS" \
        --batch_size 64 \
        --total_batch_size 512 \
        --weight_decay 0.01 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --save_every 10000 \
        --grad_clipping 1.0 \
        --save_dir "$SAVE_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --run_name "$RUN_TAG" \
        > "$RLOG" 2>&1
    rc=$?

    # 抽取 final ppl（main.py 写 SAVE_DIR/final_result.json）
    ppl="NA"
    if [ -f "$SAVE_DIR/final_result.json" ]; then
        ppl=$(grep -oE '"final_eval_ppl":[ ]*[0-9.]+' "$SAVE_DIR/final_result.json" | grep -oE '[0-9.]+$')
    fi
    echo "[sweep] LR=$LR  rc=$rc  final_ppl=${ppl:-NA}  log=$RLOG"
    RESULTS_TAGS+=("$RUN_TAG")
    RESULTS_PPLS+=("${ppl:-NA}")
done

echo ""
echo "==========================================================="
echo "[sweep] 完成。汇总（peak LR / final ppl）："
for i in "${!RESULTS_TAGS[@]}"; do
    echo "  ${RESULTS_TAGS[$i]}    ppl=${RESULTS_PPLS[$i]}"
done
echo "==========================================================="

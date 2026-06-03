#!/usr/bin/env bash
# Reset state for a fresh agent training run (new wandb run id, no resume).
# Run from project root: bash scripts/clean_run.sh
set -e

cd "$(dirname "$0")/.."

# RUN_TAG 给 save_dir 加后缀，与 agent.py 的命名保持一致（默认空=results/agent_run）
SFX=""; [ -n "${RUN_TAG:-}" ] && SFX="_${RUN_TAG}"

rm -f training_state.json training_state_temp.json
rm -f lr_command.txt lr_command_temp.txt
rm -rf "results/agent_run${SFX}/"

echo "[clean_run] reset complete (tag=${RUN_TAG:-<none>}) — next launch will start a new wandb run."

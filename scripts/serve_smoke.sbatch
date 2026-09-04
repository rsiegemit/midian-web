#!/bin/bash
#SBATCH --job-name=rte_serve_smoke
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/rte/logs/serve_smoke_%j.out
#SBATCH --error=/scratch/rte/logs/serve_smoke_%j.err
#
# One GPU, one model, one hour: the end-to-end check that env + weights + vLLM + rte.llm_client +
# LLMBackend line up before the real fleet. Defaults to the SMALLEST model in configs/models.yaml;
# override with RTE_SMOKE_MODEL.
#
# NOT gpu_test: that partition hands out A100 MIG slices, so SLURM sets CUDA_VISIBLE_DEVICES to a
# MIG UUID and vLLM 0.22.1 calls int() on it --
#   ValueError: invalid literal for int() with base 10: 'MIG-adfbd773-...'
# (vllm/platforms/cuda.py via registry.py:951, smoke job 43857516). Any whole-GPU partition works.
#
#   sbatch scripts/serve_smoke.sbatch
#   $RTE_DATA/env/rte/bin/python scripts/smoke_backend.py       # once endpoints.json lists it
set -uo pipefail

export RTE_DATA="${RTE_DATA:-/scratch/rte}"
REPO="${RTE_REPO:-$HOME/rte}"
ENV_PREFIX="$RTE_DATA/env/rte"
source "$REPO/scripts/_serve_env.sh"

MODEL="${RTE_SMOKE_MODEL:-$("$PY" -c "
import yaml
print(min(yaml.safe_load(open('$REPO/configs/models.yaml'))['models'],
          key=lambda m: m['params_b'])['id'])")}"
PORT="${RTE_SMOKE_PORT:-8011}"
HOSTNAME_FQ="$(hostname -s)"
LOG="$RTE_DATA/logs/serve_smoke_${SLURM_JOB_ID:-manual}.vllm.log"

echo "[smoke] host=$HOSTNAME_FQ job=${SLURM_JOB_ID:-manual} model=$MODEL port=$PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

cleanup() {
  "$PY" "$REPO/scripts/_register_endpoint.py" remove "$MODEL" >/dev/null 2>&1 || true
  [ -n "${VPID:-}" ] && kill "$VPID" 2>/dev/null
  sleep 3
  [ -n "${VPID:-}" ] && kill -9 "$VPID" 2>/dev/null
  true
}
trap cleanup EXIT TERM INT

serve_model "$MODEL" "$PORT" "${RTE_SMOKE_UTIL:-0.60}" "" "$LOG"
VPID=$!
if ! watch_health "$MODEL" "$PORT" "$HOSTNAME_FQ" 90; then
  echo "[smoke] tail of $LOG:"; tail -60 "$LOG"; exit 1
fi
echo "[smoke] serving until the job ends or is scancel'd"
wait "$VPID"

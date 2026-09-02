#!/bin/bash
#SBATCH --job-name=rte_serve_replica --account=kempner_sompolinsky_lab --partition=kempner_h100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/%x_%j.out --error=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/%x_%j.err
# One extra replica of a model already in the fleet (default: the 7B supervisor, the bottleneck under framework load).
# Registers under the alias "<model>#<jobid>"; clients pick uniformly among "<model>" and "<model>#*" entries.
#   RTE_REPLICA_MODEL=Qwen/Qwen2.5-7B-Instruct sbatch scripts/serve_replica.sbatch
set -uo pipefail
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"; REPO="${RTE_REPO:-$HOME/rte}"; ENV_PREFIX="$RTE_DATA/env/rte"
source "$REPO/scripts/_serve_env.sh"
MODEL="${RTE_REPLICA_MODEL:-Qwen/Qwen2.5-7B-Instruct}"; ALIAS="$MODEL#${SLURM_JOB_ID:-manual}"; PORT="${RTE_REPLICA_PORT:-8021}"
HOST="$(hostname -s)"; LOG="$RTE_DATA/logs/serve_replica_${SLURM_JOB_ID:-manual}.vllm.log"
cleanup() { "$PY" "$REPO/scripts/_register_endpoint.py" remove "$ALIAS" >/dev/null 2>&1; [ -n "${VPID:-}" ] && kill "$VPID" 2>/dev/null; sleep 3; [ -n "${VPID:-}" ] && kill -9 "$VPID" 2>/dev/null; true; }
trap cleanup EXIT TERM INT
RTE_MAX_NUM_SEQS="${RTE_MAX_NUM_SEQS:-256}" serve_model "$MODEL" "$PORT" "${RTE_REPLICA_UTIL:-0.90}" "" "$LOG"; VPID=$!
if ! watch_health "$ALIAS" "$PORT" "$HOST" 90; then tail -40 "$LOG"; exit 1; fi
wait "$VPID"

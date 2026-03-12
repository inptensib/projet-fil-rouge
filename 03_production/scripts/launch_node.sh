#!/bin/bash
#SBATCH --job-name=hydra_ddp
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem=35G
#SBATCH --output=logs/hydra_ddp_%j.log
#SBATCH --error=logs/hydra_ddp_%j.err

# =============================================================================
# HYDRA DDP -- Per-node launcher
# =============================================================================
# Launched by launch_distributed.sh for each SLURM node.
# Reads DDP configuration from .ddp_config (written by the launcher).
#
# Environment variables (from launcher):
#   GPUS_CE_NOEUD         : Number of GPUs on this node (1 or 2)
#   HYDRA_VARIANT         : Model variant (local/global/nested)
#   HYDRA_ENCODER         : Encoder name (CamemBERT/XLM-RoBERTa/mBERT)
#   FORCE_SINGLE_PROCESS  : If 1, use DataParallel instead of DDP
#   BATCH_SIZE            : Per-process batch size
# =============================================================================

GPUS_PER_NODE=${GPUS_CE_NOEUD:-2}
CONFIG_FILE="${SLURM_SUBMIT_DIR}/.ddp_config"

echo "==================================================================="
echo "HYDRA DDP -- Node $(hostname)"
echo "   GPUs on this node: ${GPUS_PER_NODE}"
echo "   Job ID: ${SLURM_JOB_ID}"
echo "   Waiting for DDP config file..."

# -- Wait for config file (written by launch_distributed.sh) ----------------
TIMEOUT=300
ELAPSED=0
while ! grep -q "READY" "$CONFIG_FILE" 2>/dev/null; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "[ERROR] Timeout (${TIMEOUT}s): DDP config file not found or incomplete."
        echo "   Expected: ${CONFIG_FILE}"
        exit 1
    fi
done
echo "   Config found after ${ELAPSED}s"

# -- Read WORLD_SIZE and RANK_OFFSET from config ----------------------------
WORLD_SIZE=$(grep "^WORLD_SIZE=" "$CONFIG_FILE" | cut -d= -f2)
RANK_OFFSET=$(grep "^${SLURM_JOB_ID} " "$CONFIG_FILE" | awk '{print $4}')

if [ -z "$WORLD_SIZE" ] || [ -z "$RANK_OFFSET" ]; then
    echo "[ERROR] Cannot read WORLD_SIZE or RANK_OFFSET from ${CONFIG_FILE}"
    echo "   SLURM_JOB_ID: ${SLURM_JOB_ID}"
    echo "   Config contents:"
    cat "$CONFIG_FILE"
    exit 1
fi

echo "   WORLD_SIZE: ${WORLD_SIZE}"
echo "   Rank offset: ${RANK_OFFSET} (ranks ${RANK_OFFSET}-$((RANK_OFFSET+GPUS_PER_NODE-1)))"
echo "   Variant: ${HYDRA_VARIANT}, Encoder: ${HYDRA_ENCODER}"
echo "   Start time: $(date)"
echo "==================================================================="

# -- Environment setup ------------------------------------------------------
VENV_PATH="${VENV_PATH:-$(cd "$(dirname "$0")/../.." && pwd)/venv}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "[WARN] No virtualenv at ${VENV_PATH}, using system Python"
fi

# HuggingFace: force offline mode (no Internet on compute nodes)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset SSL_CERT_FILE

# -- NCCL configuration for inter-node communication via RoCE (RDMA) --------
export NCCL_DEBUG=INFO

# Detect RoCE support dynamically
if [ -d /sys/class/infiniband ] && [ "$(ls /sys/class/infiniband/ 2>/dev/null)" ]; then
    echo "[NCCL] RoCE detected on $(hostname): $(ls /sys/class/infiniband/)"
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=0  # No GPU-Direct RDMA on P100
else
    echo "[NCCL] No RoCE on $(hostname) -- falling back to TCP/Socket"
    export NCCL_IB_DISABLE=1
fi

# Socket fallback: exclude virtual interfaces
export NCCL_SOCKET_IFNAME=^lo,docker0

# NVML fix: prevent nvmlDeviceGetHandleByPciBusId crash
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=LOC

# -- Launch GPU processes ---------------------------------------------------
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

NUM_PROCS=${GPUS_PER_NODE}
if [ "${FORCE_SINGLE_PROCESS:-0}" = "1" ]; then
    NUM_PROCS=1
    echo "[Mode] Single-process (DataParallel with ${GPUS_PER_NODE} GPUs)"
fi

SLURM_GPUS="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -ra GPU_IDS <<< "$SLURM_GPUS"

echo "[Launch] Starting ${NUM_PROCS} process(es) on $(hostname)..."
echo "   SLURM GPUs: ${SLURM_GPUS} (${#GPU_IDS[@]} GPU(s))"

for LOCAL_RANK in $(seq 0 $((NUM_PROCS-1))); do
    GLOBAL_RANK=$((RANK_OFFSET + LOCAL_RANK))

    if [ "${FORCE_SINGLE_PROCESS:-0}" = "1" ]; then
        GPU_DEVICE="${SLURM_GPUS}"
    else
        GPU_DEVICE="${GPU_IDS[$LOCAL_RANK]}"
    fi

    echo "   Process ${LOCAL_RANK}: GPU(s) [${GPU_DEVICE}], global_rank=${GLOBAL_RANK}"

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
    GLOBAL_RANK=$GLOBAL_RANK \
    LOCAL_RANK=$LOCAL_RANK \
    WORLD_SIZE=$WORLD_SIZE \
    PYTHONUNBUFFERED=1 \
    python -m src.main \
        --variant "${HYDRA_VARIANT:-local}" \
        --encoder "${HYDRA_ENCODER:-CamemBERT}" \
        --batch_size "${BATCH_SIZE:-128}" \
        > logs/rank_${GLOBAL_RANK}_$(hostname).log 2>&1 &
done

# Wait for all GPU processes to finish
wait

echo "==================================================================="
echo "Node $(hostname) finished at $(date)"
echo "==================================================================="

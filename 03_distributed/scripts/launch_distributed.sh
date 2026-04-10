#!/bin/bash
# =============================================================================
# HYDRA DDP -- Multi-node distributed training launcher
# =============================================================================
#
# Dynamically detects available GPU nodes and submits per-node SLURM jobs.
#
# Usage:
#   ./launch_distributed.sh                              # Defaults: local, CamemBERT, 4 nodes (P100)
#   ./launch_distributed.sh local CamemBERT 4            # 4 nodes with P100 GPUs
#   ./launch_distributed.sh local CamemBERT 1 rtx3090    # Single node with RTX 3090
#
# =============================================================================

set -euo pipefail

VARIANT=${1:-local}
ENCODER=${2:-CamemBERT}
TARGET_NODES=${3:-4}
GPU_TYPE=${4:-p100}

MIN_TOTAL_GPUS=2
QOS_MAX_GPU_PER_JOB=2
QOS_MAX_JOBS=4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "==================================================================="
echo "HYDRA DDP -- Dynamic resource detection"
echo "   Variant:      ${VARIANT}"
echo "   Encoder:      ${ENCODER}"
echo "   GPU Type:     ${GPU_TYPE}"
echo "==================================================================="

echo ""
ROCE_CACHE="${PROJECT_DIR}/.roce_nodes.cache"

declare -a ALLOC_NODES=()
declare -a ALLOC_GPUS=()

if [ "$GPU_TYPE" = "rtx3090" ]; then
    echo "[Config] Forced RTX 3090 node (node40)..."

    STATE=$(sinfo -p 3090 -N --Format="StateLong:12" --noheader \
        | grep -v "down" | grep -v "drain" | head -n 1 | awk '{print $1}')
    if [ -z "$STATE" ]; then
        echo "[ERROR] RTX 3090 node (node40) is not available."
        exit 1
    fi

    ALLOC_NODES+=("node40")
    ALLOC_GPUS+=(2)
    SELECTED_COUNT=1
    WORLD_SIZE=1
    PARTITION="3090"
    FORCE_SINGLE_PROCESS=1
    BATCH_SIZE=256
    MEM_FOR_JOB="80G"

else
    PARTITION="P100"
    FORCE_SINGLE_PROCESS=0
    BATCH_SIZE=128
    MEM_FOR_JOB="35G"
    echo "[Scan] Scanning P100 nodes (dynamic RoCE detection)..."
    echo ""

    declare -a CANDIDATE_NODES=()
    declare -a CANDIDATE_GPUS=()
    declare -a CANDIDATE_TOTAL=()
    declare -a CANDIDATE_USED=()
    declare -a CANDIDATE_STATE=()

    while IFS= read -r line; do
        NODE=$(echo "$line" | awk '{print $1}')
        TOTAL_GRES=$(echo "$line" | awk '{print $2}' | grep -oP '\d+')
        USED_GRES=$(echo "$line" | awk '{print $3}' | grep -oP '\d+')
        if [ -z "$USED_GRES" ]; then USED_GRES=0; fi
        STATE=$(echo "$line" | awk '{print $4}')
        FREE_GPUS=$((TOTAL_GRES - USED_GRES))

        if [[ "$STATE" == *"drain"* ]] || [[ "$STATE" == *"down"* ]]; then
            continue
        fi
        if [ "$FREE_GPUS" -lt 1 ]; then
            continue
        fi

        CANDIDATE_NODES+=("$NODE")
        CANDIDATE_GPUS+=("$FREE_GPUS")
        CANDIDATE_TOTAL+=("$TOTAL_GRES")
        CANDIDATE_USED+=("$USED_GRES")
        CANDIDATE_STATE+=("$STATE")

    done < <(sinfo -p P100 -N --Format="NodeList:12,Gres:8,GresUsed:16,StateLong:12" --noheader 2>/dev/null)

    echo "[RoCE] Probing ${#CANDIDATE_NODES[@]} candidate nodes..."
    if [ -f "$ROCE_CACHE" ]; then
        echo "   Cache: $(cat "$ROCE_CACHE" | tr '\n' ' ')"
    fi
    echo "   +----------+-------+---------+-------+----------+-------------------+"
    echo "   |  Node    | Total |  Used   | Free  |  State   |  RoCE             |"
    echo "   +----------+-------+---------+-------+----------+-------------------+"

    for ((idx=0; idx<${#CANDIDATE_NODES[@]}; idx++)); do
        NODE="${CANDIDATE_NODES[$idx]}"
        FREE_GPUS="${CANDIDATE_GPUS[$idx]}"
        TOTAL_GRES="${CANDIDATE_TOTAL[$idx]}"
        USED_GRES="${CANDIDATE_USED[$idx]}"
        STATE="${CANDIDATE_STATE[$idx]}"

        ROCE_DEVICE=$(ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no -o BatchMode=yes \
            "$NODE" 'ls /sys/class/infiniband/ 2>/dev/null | head -1' 2>/dev/null || true)

        if [ -n "$ROCE_DEVICE" ]; then
            ROCE_STATUS="OK ${ROCE_DEVICE} (live)"
            HAS_ROCE=true
            if ! grep -qx "$NODE" "$ROCE_CACHE" 2>/dev/null; then
                echo "$NODE" >> "$ROCE_CACHE"
            fi
        else
            if grep -qx "$NODE" "$ROCE_CACHE" 2>/dev/null; then
                ROCE_STATUS="OK (cache)"
                HAS_ROCE=true
            else
                ROCE_STATUS="absent"
                HAS_ROCE=false
            fi
        fi

        if $HAS_ROCE; then
            CAPPED_GPUS=$((FREE_GPUS > QOS_MAX_GPU_PER_JOB ? QOS_MAX_GPU_PER_JOB : FREE_GPUS))
            ALLOC_NODES+=("$NODE")
            ALLOC_GPUS+=("$CAPPED_GPUS")
        fi
        printf "   | %-8s |   %d   |    %d    |   %d   | %-8s | %-17s |\n" \
            "$NODE" "$TOTAL_GRES" "$USED_GRES" "$FREE_GPUS" "$STATE" "$ROCE_STATUS"
    done
    echo "   +----------+-------+---------+-------+----------+-------------------+"

    N_AVAILABLE=${#ALLOC_NODES[@]}
    if [ "$N_AVAILABLE" -eq 0 ]; then
        echo "[ERROR] No P100 nodes available (with RoCE)."
        exit 1
    fi

    # Sort nodes by available GPUs (descending)
    for ((i=0; i<N_AVAILABLE; i++)); do
        for ((j=i+1; j<N_AVAILABLE; j++)); do
            if [ "${ALLOC_GPUS[$j]}" -gt "${ALLOC_GPUS[$i]}" ]; then
                TMP_NODE="${ALLOC_NODES[$i]}"
                TMP_GPU="${ALLOC_GPUS[$i]}"
                ALLOC_NODES[$i]="${ALLOC_NODES[$j]}"
                ALLOC_GPUS[$i]="${ALLOC_GPUS[$j]}"
                ALLOC_NODES[$j]="$TMP_NODE"
                ALLOC_GPUS[$j]="$TMP_GPU"
            fi
        done
    done

    SELECTED_COUNT=$((N_AVAILABLE < TARGET_NODES ? N_AVAILABLE : TARGET_NODES))
    declare -a SEL_NODES=()
    declare -a SEL_GPUS=()
    WORLD_SIZE=0

    for ((i=0; i<SELECTED_COUNT; i++)); do
        SEL_NODES+=("${ALLOC_NODES[$i]}")
        SEL_GPUS+=("${ALLOC_GPUS[$i]}")
        WORLD_SIZE=$((WORLD_SIZE + ${ALLOC_GPUS[$i]}))
    done

    if [ "$WORLD_SIZE" -lt "$MIN_TOTAL_GPUS" ]; then
        echo "[ERROR] Only ${WORLD_SIZE} GPU(s) available, need at least ${MIN_TOTAL_GPUS}."
        exit 1
    fi

    ALLOC_NODES=("${SEL_NODES[@]}")
    ALLOC_GPUS=("${SEL_GPUS[@]}")
fi

echo ""
echo "==================================================================="
echo "HYDRA DDP -- Allocation plan"
echo "   Nodes:          ${SELECTED_COUNT}"
echo "   World size:     ${WORLD_SIZE} GPU(s)"
echo "   Batch/GPU:      ${BATCH_SIZE}"
echo "   Effective BS:   ${BATCH_SIZE} x ${WORLD_SIZE} = $((BATCH_SIZE * WORLD_SIZE))"
echo "   -----------------------------------------------------------"

RANK_OFFSET=0
for ((i=0; i<SELECTED_COUNT; i++)); do
    RANK_END=$((RANK_OFFSET + ${ALLOC_GPUS[$i]} - 1))
    echo "   ${ALLOC_NODES[$i]} : ${ALLOC_GPUS[$i]} GPU(s) -> ranks ${RANK_OFFSET}-${RANK_END}"
    RANK_OFFSET=$((RANK_OFFSET + ${ALLOC_GPUS[$i]}))
done
echo "==================================================================="

rm -rf "${PROJECT_DIR}/rendezvous_hydra"
CONFIG_FILE="${PROJECT_DIR}/.ddp_config"
rm -f "$CONFIG_FILE"
mkdir -p "${PROJECT_DIR}/logs"

echo ""
echo "[Submit] Phase 1: Submitting jobs (1 per node, max ${QOS_MAX_GPU_PER_JOB} GPUs/job)..."
echo ""

declare -a SUB_JOBS=()
declare -a SUB_NODES=()
declare -a SUB_GPUS=()
JOBS_SUBMITTED=0

for ((i=0; i<SELECTED_COUNT; i++)); do
    NODE="${ALLOC_NODES[$i]}"
    BATCH=${ALLOC_GPUS[$i]}

    if [ "$JOBS_SUBMITTED" -ge "$QOS_MAX_JOBS" ]; then
        break
    fi

    CPUS_FOR_JOB=$((BATCH * 5))
    [ "$CPUS_FOR_JOB" -gt 10 ] && CPUS_FOR_JOB=10

    JOB_OUTPUT=$(sbatch \
        --partition=${PARTITION} \
        --nodelist=${NODE} \
        --gpus=${BATCH} \
        --cpus-per-task=${CPUS_FOR_JOB} \
        --mem=${MEM_FOR_JOB} \
        --export=ALL,GPUS_CE_NOEUD=$BATCH,HYDRA_VARIANT=$VARIANT,HYDRA_ENCODER=$ENCODER,FORCE_SINGLE_PROCESS=$FORCE_SINGLE_PROCESS,BATCH_SIZE=$BATCH_SIZE \
        scripts/launch_node.sh 2>&1 || true)

    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$' || true)

    if [ -n "$JOB_ID" ] && [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "   ${NODE} [${BATCH} GPU(s)] -> Job $JOB_ID (submitted)"
        SUB_JOBS+=("$JOB_ID")
        SUB_NODES+=("$NODE")
        SUB_GPUS+=("$BATCH")
        JOBS_SUBMITTED=$((JOBS_SUBMITTED + 1))
    else
        echo "   ${NODE} [${BATCH} GPU(s)] -> submission failed"
    fi
done

if [ ${#SUB_JOBS[@]} -eq 0 ]; then
    echo "[ERROR] No jobs submitted."
    exit 1
fi

POLL_INTERVAL=5
POLL_TIMEOUT=90

echo ""
echo "[Wait] Phase 2: Waiting for jobs to start (poll ${POLL_INTERVAL}s, timeout ${POLL_TIMEOUT}s)..."

ELAPSED=0
while [ "$ELAPSED" -lt "$POLL_TIMEOUT" ]; do
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))

    N_PENDING=0
    for ((i=0; i<${#SUB_JOBS[@]}; i++)); do
        STATE=$(squeue -j "${SUB_JOBS[$i]}" --noheader -o "%T" 2>/dev/null || true)
        if [ "$STATE" = "PENDING" ]; then
            N_PENDING=$((N_PENDING + 1))
        fi
    done

    if [ "$N_PENDING" -eq 0 ]; then
        echo "   All jobs started after ${ELAPSED}s"
        break
    fi
    echo "   ${ELAPSED}s/${POLL_TIMEOUT}s -- ${N_PENDING} job(s) still PENDING..."
done

echo ""
declare -a STARTED_JOBS=()
declare -a STARTED_NODES=()
declare -a STARTED_GPUS=()
declare -a PENDING_JOBS=()

for ((i=0; i<${#SUB_JOBS[@]}; i++)); do
    JID="${SUB_JOBS[$i]}"
    NODE="${SUB_NODES[$i]}"
    GPUS="${SUB_GPUS[$i]}"

    STATE=$(squeue -j "$JID" --noheader -o "%T" 2>/dev/null || true)

    if [ "$STATE" = "RUNNING" ]; then
        echo "   Job $JID (${NODE}, ${GPUS} GPU) -> RUNNING"
        STARTED_JOBS+=("$JID")
        STARTED_NODES+=("$NODE")
        STARTED_GPUS+=("$GPUS")
    else
        echo "   Job $JID (${NODE}, ${GPUS} GPU) -> ${STATE:-ENDED} (will be cancelled)"
        PENDING_JOBS+=("$JID")
    fi
done

if [ ${#PENDING_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "[Cleanup] Cancelling ${#PENDING_JOBS[@]} non-started job(s)..."
    for JID in "${PENDING_JOBS[@]}"; do
        scancel "$JID" 2>/dev/null || true
    done
fi

ACTUAL_WORLD_SIZE=0
ACTUAL_RANKS=0
for g in "${STARTED_GPUS[@]}"; do
    ACTUAL_WORLD_SIZE=$((ACTUAL_WORLD_SIZE + g))
    if [ "${FORCE_SINGLE_PROCESS:-0}" = "1" ]; then
        ACTUAL_RANKS=$((ACTUAL_RANKS + 1))
    else
        ACTUAL_RANKS=$((ACTUAL_RANKS + g))
    fi
done

if [ "$ACTUAL_WORLD_SIZE" -lt "$MIN_TOTAL_GPUS" ]; then
    echo "[ERROR] Only ${ACTUAL_WORLD_SIZE} GPU(s) running, need at least ${MIN_TOTAL_GPUS}."
    for JID in "${STARTED_JOBS[@]}"; do
        scancel "$JID" 2>/dev/null || true
    done
    rm -f "$CONFIG_FILE"
    exit 1
fi

echo ""
echo "[Config] Writing DDP config: ${CONFIG_FILE}"
{
    echo "WORLD_SIZE=${ACTUAL_RANKS}"
    RANK_OFFSET=0
    for ((i=0; i<${#STARTED_JOBS[@]}; i++)); do
        echo "${STARTED_JOBS[$i]} ${STARTED_NODES[$i]} ${STARTED_GPUS[$i]} ${RANK_OFFSET}"
        if [ "${FORCE_SINGLE_PROCESS:-0}" = "1" ]; then
            RANK_OFFSET=$((RANK_OFFSET + 1))
        else
            RANK_OFFSET=$((RANK_OFFSET + ${STARTED_GPUS[$i]}))
        fi
    done
    echo "READY"
} > "$CONFIG_FILE"
echo "   Config written successfully."

echo ""
echo "==================================================================="
echo "HYDRA DDP -- Summary"
echo "   Nodes started:  ${#STARTED_JOBS[@]}"
echo "   World size:     ${ACTUAL_RANKS} rank(s)"
echo "   GPUs total:     ${ACTUAL_WORLD_SIZE}"
echo "   Batch per rank: ${BATCH_SIZE}"
echo "   Effective BS:   ${BATCH_SIZE} x ${ACTUAL_RANKS} = $((BATCH_SIZE * ACTUAL_RANKS))"
echo "==================================================================="

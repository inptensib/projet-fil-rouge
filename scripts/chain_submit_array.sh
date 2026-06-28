#!/bin/bash
# Soumet N job arrays chaînés avec dépendances automatiques.
# Chaque array contient 3 tâches (local, nested, global).
#
# Usage:
#   ./scripts/chain_submit_array.sh [n_chains] [config]
#   ./scripts/chain_submit_array.sh 3 configs/hydra/hierarchy.yaml

set -euo pipefail

N_CHAINS=${1:-3}
CONFIG=${2:-configs/hydra/default.yaml}

echo "Soumission de $N_CHAINS job arrays (3 variantes chacun, config: $CONFIG)"
echo "=============================================="

PREV_JOB=""
for i in $(seq 1 "$N_CHAINS"); do
    if [ -z "$PREV_JOB" ]; then
        JOB=$(sbatch --parsable scripts/slurm/run_train_array.slurm "$CONFIG")
    else
        JOB=$(sbatch --parsable --dependency=afterany:"$PREV_JOB" scripts/slurm/run_train_array.slurm "$CONFIG")
    fi
    echo "  Chain $i/$N_CHAINS : job array $JOB (depends=${PREV_JOB:-none})"
    PREV_JOB=$JOB
done

echo "=============================================="
echo "$N_CHAINS arrays soumis (= $((N_CHAINS * 3)) tâches max)"
echo "Chaque array entraîne local + nested + global en parallèle."
echo "squeue -u \$USER pour suivre."

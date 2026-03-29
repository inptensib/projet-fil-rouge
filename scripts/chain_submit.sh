#!/bin/bash
# Soumet une chaine de N jobs SLURM avec dependances automatiques.
# Chaque job reprend le training la ou le precedent s'est arrete.
#
# Usage:
#   ./scripts/chain_submit.sh <variant> [n_jobs]
#   ./scripts/chain_submit.sh local 6
#   ./scripts/chain_submit.sh nested 6
#   ./scripts/chain_submit.sh global 6

set -euo pipefail

VARIANT=${1:?"Usage: $0 <variant> [n_jobs]  (variant: local|global|nested)"}
N_JOBS=${2:-6}

echo "Soumission de $N_JOBS jobs pour HYDRA $VARIANT"
echo "=============================================="

PREV_JOB=""
for i in $(seq 1 "$N_JOBS"); do
    if [ -z "$PREV_JOB" ]; then
        JOB=$(sbatch --parsable scripts/run_train_p100.slurm "$VARIANT")
    else
        JOB=$(sbatch --parsable --dependency=afterany:"$PREV_JOB" scripts/run_train_p100.slurm "$VARIANT")
    fi
    echo "  Job $i/$N_JOBS : $JOB (depends=${PREV_JOB:-none})"
    PREV_JOB=$JOB
done

echo "=============================================="
echo "Chaine de $N_JOBS jobs soumise pour $VARIANT"
echo "Le training reprendra automatiquement apres chaque kill SLURM."
echo "Utilisez 'squeue -u \$USER' pour suivre la progression."

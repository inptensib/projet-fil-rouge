#!/bin/bash
# Soumet une chaine de N jobs SLURM avec dependances automatiques.
# Chaque job reprend le training la ou le precedent s'est arrete.
#
# Usage:
#   ./scripts/chain_submit.sh <variant> [n_jobs] [config]
#   ./scripts/chain_submit.sh local 6
#   ./scripts/chain_submit.sh local 3 02_hydra/configs/hierarchy.yaml

set -euo pipefail

VARIANT=${1:?"Usage: $0 <variant> [n_jobs] [config]  (variant: local|global|nested)"}
N_JOBS=${2:-6}
CONFIG=${3:-02_hydra/configs/default.yaml}

echo "Soumission de $N_JOBS jobs pour HYDRA $VARIANT (config: $CONFIG)"
echo "=============================================="

PREV_JOB=""
for i in $(seq 1 "$N_JOBS"); do
    if [ -z "$PREV_JOB" ]; then
        JOB=$(sbatch --parsable --export=ALL,HYDRA_VARIANT="$VARIANT",HYDRA_CONFIG="$CONFIG" scripts/run_train_p100.slurm "$VARIANT" "$CONFIG")
    else
        JOB=$(sbatch --parsable --dependency=afterany:"$PREV_JOB" --export=ALL,HYDRA_VARIANT="$VARIANT",HYDRA_CONFIG="$CONFIG" scripts/run_train_p100.slurm "$VARIANT" "$CONFIG")
    fi
    echo "  Job $i/$N_JOBS : $JOB (depends=${PREV_JOB:-none})"
    PREV_JOB=$JOB
done

echo "=============================================="
echo "Chaine de $N_JOBS jobs soumise pour $VARIANT"
echo "Le training reprendra automatiquement apres chaque kill SLURM."
echo "Utilisez 'squeue -u \$USER' pour suivre la progression."

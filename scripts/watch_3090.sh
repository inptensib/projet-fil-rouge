#!/bin/bash
# Surveille la partition 3090 (node40) et migre les jobs HYDRA P100 → 3090
# quand le nœud se libère. Le training reprend depuis le dernier checkpoint.
#
# Usage:
#   nohup bash scripts/watch_3090.sh > watch_3090.log 2>&1 &

set -uo pipefail

cd /home/infres/tandriam-25/fil_rouge/rakuten-classification
source /home/infres/tandriam-25/fil_rouge/venv/bin/activate

CONFIG="02_hydra/configs/hierarchy.yaml"
SAVE_DIR="models_hierarchy"
CHECK_INTERVAL=300  # 5 minutes

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Trouver la variante P100 la plus en retard (global_step le plus bas)
find_slowest_p100_variant() {
    local best_variant=""
    local best_step=999999999

    for variant in local nested global; do
        # Vérifier si un job P100 tourne pour cette variante
        local running=$(squeue -u "$USER" -p P100 --format="%i %o" --noheader 2>/dev/null \
            | grep "run_train_p100.slurm" | head -1)
        if [ -z "$running" ]; then
            continue
        fi

        # Vérifier le checkpoint
        local ckpt="${SAVE_DIR}/HYDRA_${variant}_CamemBERT/checkpoint_latest.pt"
        local complete="${SAVE_DIR}/HYDRA_${variant}_CamemBERT/training_complete.txt"

        # Skip si training déjà terminé
        if [ -f "$complete" ]; then
            continue
        fi

        local step=0
        if [ -f "$ckpt" ]; then
            step=$(python3 -c "
import torch
ckpt = torch.load('$ckpt', map_location='cpu', weights_only=False)
print(ckpt.get('global_step', 0))
" 2>/dev/null || echo 0)
        fi

        if [ "$step" -lt "$best_step" ]; then
            best_step=$step
            best_variant=$variant
        fi
    done

    echo "$best_variant"
}

# Annuler les jobs P100 d'une variante (job + dépendances)
cancel_p100_variant() {
    local variant=$1
    local job_ids=$(squeue -u "$USER" -p P100,3090,P100 --format="%i" --noheader 2>/dev/null)

    # Trouver les jobs de cette variante en vérifiant les logs
    for jid in $job_ids; do
        local logfile="slurm-hydra-p100-${jid}.out"
        if [ -f "$logfile" ] && grep -q "Variant: ${variant}" "$logfile" 2>/dev/null; then
            log "  Annulation job $jid (${variant})"
            scancel "$jid" 2>/dev/null
        fi
    done

    # Annuler aussi les jobs PENDING de cette chaîne
    local pending=$(squeue -u "$USER" --format="%i %T" --noheader 2>/dev/null | grep PENDING | awk '{print $1}')
    for jid in $pending; do
        local state=$(scontrol show job "$jid" 2>/dev/null | grep -oP 'Dependency=afterany:\K\d+')
        if [ -n "$state" ]; then
            # Ce job dépend d'un autre — vérifier si c'est notre variante
            local parent_log="slurm-hydra-p100-${state}.out"
            if [ -f "$parent_log" ] && grep -q "Variant: ${variant}" "$parent_log" 2>/dev/null; then
                log "  Annulation job chaîné $jid"
                scancel "$jid" 2>/dev/null
            fi
        fi
    done
}

log "=== Démarrage watch_3090 ==="
log "Surveille node40 toutes les ${CHECK_INTERVAL}s"

while true; do
    # Vérifier si il reste des jobs P100 HYDRA
    p100_hydra=$(squeue -u "$USER" -p P100 --format="%i" --noheader 2>/dev/null | wc -l)
    if [ "$p100_hydra" -eq 0 ]; then
        log "Plus de jobs HYDRA sur P100. Fin du watcher."
        break
    fi

    # Compter les GPUs utilisées sur node40 (on a besoin de 2, il y en a 3 au total)
    gpus_used=$(squeue -w node40 --format="%b" --noheader 2>/dev/null \
        | grep -oP 'gpu:\K\d+' | awk '{s+=$1}END{print s+0}')
    gpus_free=$((3 - gpus_used))

    if [ "$gpus_free" -ge 2 ]; then
        log "*** node40 : ${gpus_free} GPU(s) libres (${gpus_used}/3 utilisées) ! Migration en cours..."

        variant=$(find_slowest_p100_variant)
        if [ -z "$variant" ]; then
            log "Aucune variante P100 à migrer (toutes terminées ?)"
            sleep "$CHECK_INTERVAL"
            continue
        fi

        log "  Variante sélectionnée : $variant (la plus en retard)"

        # Annuler les jobs P100 de cette variante
        cancel_p100_variant "$variant"
        sleep 5  # Laisser SLURM traiter les annulations

        # Soumettre sur 3090 uniquement
        log "  Soumission sur 3090..."
        # Forcer partition 3090 via variable d'environnement
        PREV_JOB=""
        for i in 1 2; do
            if [ -z "$PREV_JOB" ]; then
                JOB=$(sbatch --parsable --partition=3090 \
                    --export=ALL,HYDRA_VARIANT="$variant",HYDRA_CONFIG="$CONFIG" \
                    scripts/run_train_p100.slurm "$variant" "$CONFIG")
            else
                JOB=$(sbatch --parsable --partition=3090 --dependency=afterany:"$PREV_JOB" \
                    --export=ALL,HYDRA_VARIANT="$variant",HYDRA_CONFIG="$CONFIG" \
                    scripts/run_train_p100.slurm "$variant" "$CONFIG")
            fi
            log "  Job $i/2 : $JOB (partition=3090)"
            PREV_JOB=$JOB
        done

        log "  Migration $variant : P100 → 3090 terminée"

        # Attendre un peu avant de vérifier à nouveau
        sleep 60
    else
        sleep "$CHECK_INTERVAL"
    fi
done

log "=== watch_3090 terminé ==="

#!/usr/bin/env bash
# One training run of train.py on one GB200 GPU, inside our image (surrogate_model_deepXDE.def).
# Submitted by dalia.sh from the synced folder (the job runs there); by hand, from that folder:
#   sbatch -o slurm_logs/%x_%j.out dalia/train.sbatch --model-file models/torch_mlp.py
# As a job array, every task reads its train.py arguments from one line of a sweep file:
#   sbatch --array=1-6 -o slurm_logs/%x_%A_%a.out dalia/train.sbatch --sweep dalia/sweeps/x.txt
#SBATCH --job-name=pallas
#SBATCH --reservation=pdl18409_oct2026
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --time=02:00:00

set -euo pipefail
ulimit -s 8192
cd "$SLURM_SUBMIT_DIR"
source dalia/env.sh

if [[ "${1:-}" == "--sweep" ]]; then
    sweep=$2
    shift 2
    extra=("$@")                                    # arguments common to every line
    line=$(grep -vE '^[[:space:]]*(#|$)' "$sweep" | sed -n "${SLURM_ARRAY_TASK_ID}p")
    [[ -n "$line" ]] || { echo "no line ${SLURM_ARRAY_TASK_ID} in $sweep" >&2; exit 1; }
    eval "set -- $line"
    set -- "$@" ${extra[@]+"${extra[@]}"}
fi

# The rows come from data/fv_rows/ (exported by dalia.sh): the compute nodes' proxy refuses the
# Feature Query Service. Everything else (statistics, registry, W&B) goes over HTTPS as usual.
set -- --rows cache "$@"
echo "job ${SLURM_JOB_ID} on $(hostname), $(date -Is)"
echo "train.py $*"
srun apptainer exec --nv --bind "$BIND" "$CONTAINER" \
    python3 -u "$PALLAS_DIR/train.py" "$@"

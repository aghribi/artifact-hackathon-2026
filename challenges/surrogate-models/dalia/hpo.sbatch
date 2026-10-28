#!/usr/bin/env bash
# A hyperparameter search of MO_DeepOnet (hpo.py) on two GB200 GPUs, inside our image.
# Submitted by `dalia.sh hpo` from the synced folder; by hand, from that folder:
#   sbatch -o slurm_logs/%x_%j.out dalia/hpo.sbatch --study deeponet_ood_v1
# hpo.py starts --workers-per-gpu processes on each GPU Slurm gives the job and, when the search
# is over, retrains and registers only the best model. 5 h covers the default 3 h of search, the
# trials still running at its end, and the final fit.
#SBATCH --job-name=pallas-hpo
#SBATCH --reservation=pdl18409_oct2026
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=72
#SBATCH --time=05:00:00

set -euo pipefail
ulimit -s 8192
cd "$SLURM_SUBMIT_DIR"
source dalia/env.sh

# Rows from data/fv_rows/ (exported by dalia.sh): the compute nodes' proxy refuses the Feature
# Query Service. Statistics and the registry go over HTTPS as usual.
set -- --rows cache "$@"
echo "job ${SLURM_JOB_ID} on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-none}, $(date -Is)"
echo "hpo.py $*"
srun apptainer exec --nv --bind "$BIND" "$CONTAINER" \
    python3 -u "$PALLAS_DIR/hpo.py" "$@"

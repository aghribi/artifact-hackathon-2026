#!/usr/bin/env bash
#SBATCH --job-name=ess_pm_bpm_pretrip
#SBATCH --reservation=pdl18409_oct2026
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --time=00:30:00

# CPU only: no GPU is requested, so none is taken from the teams.
#   sbatch ess_pm_bpm_pretrip.sh
# Reads out_dd/mps_transitions.parquet next to the script, output goes to out_pm_bpm/.
# Submit from the folder holding ess_pm_bpm_pretrip.py, ess_archiver_pretrip.py and an existing slurm_logs/.

set -euo pipefail

PROJECT_DIR=/lustre/work/pdl18409/$USER
export APPTAINER_CACHEDIR=$PROJECT_DIR/cache_apptainer
mkdir -p "$APPTAINER_CACHEDIR"
# One process per core: without this every worker starts its own pool of math threads and the node thrashes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
srun apptainer exec \
  --bind /lustre/work/pdl18409 \
  /lustre/work/pdl18409/shared/containers/anomaly_detections.sif \
  python3 ess_pm_bpm_pretrip.py /lustre/work/pdl18409/shared/data/anomaly-detection/raw \
  "$PWD/out_dd/mps_transitions.parquet" "$PWD/out_pm_bpm"

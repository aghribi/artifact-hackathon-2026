#!/usr/bin/env bash
#SBATCH --job-name=vllm_download
#SBATCH --reservation=pdl18409_oct2026
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00

# Downloads a model into the Hugging Face cache on Lustre. No GPU: the serve job
# should not hold GPUs while 198 GB come down.
# Usage: sbatch vllm_download.sh [model id]

set -euo pipefail

MODEL="${1:-RedHatAI/GLM-5.3-Flash-NVFP4}"
umask 007
DIR=/lustre/work/pdl18409/shared/vllm
export HF_HOME=$DIR/cache_huggingface
SIF=$DIR/vllm-glm53-flash.sif

apptainer exec --bind /lustre "$SIF" python3 -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], max_workers=16))
" "$MODEL"

du -sh "$HF_HOME"

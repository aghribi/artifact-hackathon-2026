#!/usr/bin/env bash
#SBATCH --job-name=vllm_serve
#SBATCH --reservation=pdl18409_oct2026
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00

# Serves GLM-5.3-Flash on two GB200 behind an API key, on the cluster network only.
# The default is the NVFP4 checkpoint: the FP8 one (zai-org/GLM-5.3-Flash) takes 158 of the 184 GiB
# of each GPU and runs out of memory at startup.
# Usage: sbatch [--time=HH:MM:SS] vllm_serve.sh [model id]

set -euo pipefail

# Everything lives in the group folder, read and write for the hackathon group.
umask 007
DIR=/lustre/work/pdl18409/shared/vllm
export HF_HOME=$DIR/cache_huggingface
export HF_HUB_OFFLINE=1
MODEL="${1:-RedHatAI/GLM-5.3-Flash-NVFP4}"
SIF=$DIR/vllm-glm53-flash.sif

# Hackathon members read the node name and the key from the group folder.
# The node name changes with each job; the key is kept across jobs.
[ -f "$DIR/api_key" ] || openssl rand -hex 24 > "$DIR/api_key"
hostname -s > "$DIR/node"
export VLLM_API_KEY=$(cat "$DIR/api_key")

# The default 2 MB stack segfaults NCCL on this machine (IDRIS DALIA FAQ).
ulimit -s 8192

apptainer exec --nv --bind /lustre "$SIF" \
  vllm serve "$MODEL" \
    --served-model-name glm-5.3-flash \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --tool-call-parser glm47 \
    --reasoning-parser glm47 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 --port 8000

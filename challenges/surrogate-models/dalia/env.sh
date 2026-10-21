# Paths and caches on DALIA, sourced by train.sbatch and dalia.sh's remote commands.
# Everything heavy goes to Lustre: $HOME has a 3 GB / 12,000-inode quota (lecture 03).

export PROJECT_DIR=${PROJECT_DIR:-/lustre/work/pdl18409/$USER}
export PALLAS_DIR=${PALLAS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export CONTAINER=${CONTAINER:-$PROJECT_DIR/containers/surrogate_model_deepXDE.sif}   # dalia.sh build
export BIND=${BIND:-/lustre/work/pdl18409}            # same paths inside and outside the container

export APPTAINER_CACHEDIR=$PROJECT_DIR/cache_apptainer
export PIP_CACHE_DIR=$PROJECT_DIR/cache_pip
export XDG_CACHE_HOME=$PROJECT_DIR/cache
export MPLCONFIGDIR=$PROJECT_DIR/cache/matplotlib
export HF_HOME=$PROJECT_DIR/cache_huggingface
export TRITON_HOME=$PROJECT_DIR/cache_triton
export DDE_BACKEND=pytorch
mkdir -p "$APPTAINER_CACHEDIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

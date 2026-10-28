#!/usr/bin/env bash
# Drive training on DALIA from this machine: sync the code, queue jobs, watch them, fetch results.
#
#   dalia/dalia.sh login                    open the shared SSH connection (CC-IN2P3 password, once);
#                                           DALIA_JUMP=user@cca012.in2p3.fr if cca.in2p3.fr hangs
#   dalia/dalia.sh sync                     copy code, models/, the data pack and .env to DALIA
#   dalia/dalia.sh build                    sync, then build our image (surrogate_model_deepXDE.def) on a compute node
#   dalia/dalia.sh submit <train.py args>   sync, then queue one training job (1 GPU)
#   dalia/dalia.sh sweep <file> [args]      sync, then queue one job per line of <file> (job array)
#   dalia/dalia.sh hpo [hpo.py args]        sync, then queue an Optuna search of MO_DeepOnet on
#                                           2 GPUs; only its best model is registered
#   dalia/dalia.sh status                   your jobs in the queue
#   dalia/dalia.sh logs [job] [-f]          a job's log (the latest when no job is given)
#   dalia/dalia.sh fetch                    copy submissions/, outputs/runs/ and outputs/hpo/ back
#   dalia/dalia.sh cancel <job>             cancel a job
#   dalia/dalia.sh shell                    interactive GPU shell inside the container (debugging)
#
# --dry-run before the command prints what would run on DALIA instead of running it.
# Environment: DALIA_HOST (ssh alias, default dalia), DALIA_DIR (remote folder, default
# /lustre/work/pdl18409/<user>/pallas), DALIA_RESERVATION, DALIA_SBATCH (extra sbatch flags,
# e.g. "--time=06:00:00"), DALIA_ARRAY_MAX (jobs of a sweep running at once, default 4).
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)                  # challenges/surrogate-models
HOST=${DALIA_HOST:-dalia}
DALIA_USER=${DALIA_USER:-$(ssh -G "$HOST" 2>/dev/null | awk '/^user /{print $2}')}
REMOTE=${DALIA_DIR:-/lustre/work/pdl18409/$DALIA_USER/pallas}
RESERVATION=${DALIA_RESERVATION:-pdl18409_oct2026}
ARRAY_MAX=${DALIA_ARRAY_MAX:-4}
# One SSH connection, opened by `login`, reused by every command: the jump host asks a password.
SSH_OPTS=(-x -o "ControlPath=$HOME/.ssh/cm-dalia-%r@%h-%p" -o ControlMaster=no -o BatchMode=yes)
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi

die() { echo "dalia.sh: $*" >&2; exit 1; }

connected() {
    [[ $DRY == 1 ]] || ssh "${SSH_OPTS[@]}" -O check "$HOST" 2>/dev/null \
        || die "no open connection to $HOST: run dalia/dalia.sh login first"
}

# Run a command on the login node, in $REMOTE, with the login environment (sbatch on PATH).
remote() {
    local command="cd $REMOTE && source dalia/env.sh && $1"
    if [[ $DRY == 1 ]]; then echo "[on $HOST] $command"; return; fi
    connected
    ssh "${SSH_OPTS[@]}" ${TTY:+-t} "$HOST" "bash -lc $(printf '%q' "$command")"
}

quote() { printf '%q ' "$@"; }

env_file() {    # the nearest .env, as team_wandb.load_env finds it
    local folder=$HERE
    while [[ "$folder" != / ]]; do
        [[ -f "$folder/.env" ]] && { echo "$folder/.env"; return; }
        folder=$(dirname "$folder")
    done
    die "no .env in $HERE or above it (HOPSWORKS_*, WANDB_API_KEY)"
}

sync_code() {
    local excludes=(--exclude outputs/ --exclude submissions/ --exclude wandb/
                    --exclude slurm_logs/ --exclude __pycache__/ --exclude .ipynb_checkpoints/
                    --exclude .DS_Store --exclude '*.ipynb' --exclude '*.pptx'
                    --exclude presentation/ --exclude data/sample/ --exclude .env --exclude .venv/)
    local envfile
    envfile=$(env_file)
    if [[ $DRY == 1 ]]; then
        echo "[local] rsync -az ${excludes[*]} $HERE/ $HOST:$REMOTE/"
        echo "[local] copy the HOPSWORKS_* lines of $envfile -> $HOST:$REMOTE/.env (mode 600)"
        return
    fi
    connected
    ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p $REMOTE"
    rsync -az "${excludes[@]}" -e "ssh ${SSH_OPTS[*]}" "$HERE/" "$HOST:$REMOTE/"
    # Only the Hopsworks key: nothing on DALIA logs to W&B (the organisers score the CSV first)
    grep -E '^[[:space:]]*HOPSWORKS_' "$envfile" \
        | ssh "${SSH_OPTS[@]}" "$HOST" "umask 077 && cat > $REMOTE/.env"
    echo "synced $HERE -> $HOST:$REMOTE"
}

local_python() {    # $PYTHON, else the nearest .venv in this folder or above it, else python3
    [[ -n "${PYTHON:-}" ]] && { echo "$PYTHON"; return; }
    local folder=$HERE
    while [[ "$folder" != / ]]; do
        [[ -x "$folder/.venv/bin/python" ]] && { echo "$folder/.venv/bin/python"; return; }
        folder=$(dirname "$folder")
    done
    echo python3
}

# DALIA's compute nodes cannot read the feature store (their proxy refuses the Feature Query
# Service), so the rows of the view a job uses are exported here first, into data/fv_rows/,
# and synced with the rest. Only once per view version: an existing export is kept.
export_rows() {         # EXPORT_WITH=hpo.py for hpo.py's arguments
    local script=${EXPORT_WITH:-train.py}
    if [[ $DRY == 1 ]]; then echo "[local] $(local_python) $script --rows export $*"; return; fi
    "$(local_python)" "$HERE/$script" --rows export "$@" 2>&1 | grep -E '^\[|Error|error' || true
    [[ ${PIPESTATUS[0]} == 0 ]] || die "exporting the rows failed: $script --rows export $*"
}

export_sweep_rows() {   # every line of a sweep file, with the arguments common to all of them
    local file=$1 line
    shift
    local extra=("$@")
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        eval "set -- $line"
        export_rows "$@" ${extra[@]+"${extra[@]}"}
    done < "$file"
}

job_name() {    # pallas-<model file stem>, so squeue tells the jobs apart
    local previous=""
    for arg in "$@"; do
        [[ "$previous" == --model-file ]] && { basename "$arg" .py; return; }
        previous=$arg
    done
    echo run
}

case "${1:-help}" in
    login)
        # Keep-alives: an idle connection is otherwise dropped on the way (broken pipe).
        # DALIA_JUMP=user@cca012.in2p3.fr picks a CC-IN2P3 login node when the one that
        # cca.in2p3.fr points to does not answer.
        ssh -x -o "ControlPath=$HOME/.ssh/cm-dalia-%r@%h-%p" -o ControlMaster=yes -o ControlPersist=12h \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
            ${DALIA_JUMP:+-J "$DALIA_JUMP"} -fN "$HOST"
        echo "connected to $HOST for 12 h (dalia/dalia.sh logout to close)" ;;
    logout)
        ssh "${SSH_OPTS[@]}" -O exit "$HOST" ;;
    sync)
        sync_code ;;
    build)
        # CPU only (no GPU held while it builds); built aside and then moved into place, so jobs
        # running on the old image keep it. `check` then tries it on a GPU.
        sync_code
        # The output is also kept in slurm_logs/build_<time>.log, to read after the fact.
        remote "mkdir -p \$(dirname \$CONTAINER) slurm_logs && srun --reservation=$RESERVATION \
--nodes=1 --ntasks-per-node=1 --cpus-per-task=36 --time=01:00:00 bash -c \
'apptainer build --fakeroot --force \$CONTAINER.new dalia/surrogate_model_deepXDE.def \
&& apptainer exec --bind \$BIND \$CONTAINER.new python3 dalia/check_image.py \
&& mv \$CONTAINER.new \$CONTAINER && echo built \$CONTAINER' 2>&1 \
| tee slurm_logs/build_\$(date +%Y%m%d-%H%M%S).log; exit \${PIPESTATUS[0]}" ;;
    submit)
        shift
        [[ $# -gt 0 ]] || die "submit needs train.py arguments, e.g. --model-file models/torch_mlp.py"
        export_rows "$@"
        sync_code
        remote "mkdir -p slurm_logs && sbatch --parsable --reservation=$RESERVATION \
-J pallas-$(job_name "$@") -o 'slurm_logs/%x_%j.out' ${DALIA_SBATCH:-} \
dalia/train.sbatch $(quote "$@")" ;;
    sweep)
        shift
        [[ $# -gt 0 && -f "$1" ]] || die "sweep needs a file of train.py argument lines"
        file=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
        file=${file#"$HERE"/}
        shift
        n=$(grep -cvE '^[[:space:]]*(#|$)' "$HERE/$file")
        export_sweep_rows "$HERE/$file" "$@"
        sync_code
        remote "mkdir -p slurm_logs && sbatch --parsable --reservation=$RESERVATION \
--array=1-$n%$ARRAY_MAX -J pallas-$(basename "$file" .txt) -o 'slurm_logs/%x_%A_%a.out' \
${DALIA_SBATCH:-} dalia/train.sbatch --sweep $(quote "$file" "$@")" ;;
    hpo)
        shift
        EXPORT_WITH=hpo.py export_rows "$@"
        sync_code
        remote "mkdir -p slurm_logs && sbatch --parsable --reservation=$RESERVATION \
-o 'slurm_logs/%x_%j.out' ${DALIA_SBATCH:-} dalia/hpo.sbatch $(quote "$@")" ;;
    status)
        remote "squeue -u \$USER -o '%.12i %.24j %.8T %.10M %.10l %R'" ;;
    check)      # the image's versions and GPU, on a compute node
        remote "srun --reservation=$RESERVATION --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 \
--cpus-per-task=36 --time=00:10:00 apptainer exec --nv --bind \$BIND \$CONTAINER \
python3 dalia/check_image.py" ;;
    logs)
        shift
        follow=""; job=""
        for arg in "$@"; do [[ "$arg" == -f ]] && follow=1 || job=$arg; done
        if [[ -n "$job" ]]; then pattern="slurm_logs/*_${job}*.out"; count=20   # an array: all tasks
        else pattern="slurm_logs/*.out"; count=1; fi
        files="\$(ls -t $pattern | head -$count)"
        if [[ -n "$follow" ]]; then TTY=1 remote "tail -n 50 -f $files"
        else remote "tail -n 80 $files"; fi ;;
    fetch)
        for folder in submissions outputs/runs outputs/hpo; do
            if [[ $DRY == 1 ]]; then echo "[local] rsync -az $HOST:$REMOTE/$folder/ $HERE/$folder/"
                continue; fi
            connected
            mkdir -p "$HERE/$folder"
            rsync -az --exclude data.pkl -e "ssh ${SSH_OPTS[*]}" "$HOST:$REMOTE/$folder/" "$HERE/$folder/"
        done
        [[ $DRY == 1 ]] || python3 "$HERE/dalia/runs_table.py" "$HERE/outputs/runs" ;;
    cancel)
        shift
        remote "scancel $(quote "$@")" ;;
    shell)
        TTY=1 remote "srun --reservation=$RESERVATION --nodes=1 --ntasks-per-node=1 \
--gpus-per-node=1 --cpus-per-task=36 --time=01:00:00 --pty apptainer shell --nv \
--bind \$BIND \$CONTAINER" ;;
    *)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
esac

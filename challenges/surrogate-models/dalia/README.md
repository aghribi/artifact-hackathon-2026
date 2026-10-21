# Training on DALIA

`train.py` (one folder up) is `surrogate_pipeline.ipynb` without the notebook: feature view →
split → fit → holdout scores → bundle → registry → submission CSV. `dalia.sh` runs it on
DALIA's GB200 GPUs from your laptop, one job per GPU, following lecture 03 (Apptainer
container, `pdl18409_oct2026` reservation, caches on Lustre). The image, `surrogate_model_deepXDE.def`, is
the shared `surrogate_models.sif` plus `hopsworks[python]==5.1.1`, `wandb==0.30.0` and
`deepxde==1.15.0`, the versions of `requirements.txt`.

| File | Runs where | What it does |
|---|---|---|
| `dalia.sh` | your laptop | sync, build, check, submit, sweep, status, logs, fetch, cancel, shell |
| `train.sbatch` | DALIA, Slurm | one `train.py` run in `surrogate_model_deepXDE.sif`, 1 GPU, 36 cores, 2 h |
| `surrogate_model_deepXDE.def` | DALIA compute node | the image, built by `dalia.sh build` into `$PROJECT_DIR/containers/` |
| `check_image.py` | inside the image | versions and GPU, after a build or with `dalia.sh check` |
| `env.sh` | DALIA | paths and caches (everything on `/lustre/work/pdl18409/$USER`) |
| `sweeps/*.txt` | — | one line of `train.py` arguments per job of a job array |
| `hpo.sbatch` | DALIA, Slurm | `hpo.py`: an Optuna search of MO_DeepOnet on 2 GPUs, 72 cores, 5 h |

## Once

```bash
dalia/dalia.sh login     # CC-IN2P3 password once; the connection stays open 12 h
dalia/dalia.sh build     # copies code, data pack and .env, builds the image on a compute node
```

## Every run

```bash
dalia/dalia.sh submit --split ood_density --model-file models/MO_DeepOnet.py \
    --params '{"ensemble": 4}' --register
dalia/dalia.sh sweep dalia/sweeps/deeponet_ood.txt           # 6 jobs, at most 4 at once
dalia/dalia.sh status
dalia/dalia.sh logs <job> -f
dalia/dalia.sh fetch     # submissions/ and outputs/runs/ back here, plus a table of the runs
```

## Hyperparameter search (OOD trajectory)

```bash
dalia/dalia.sh hpo --study deeponet_ood_v1    # 60 trials or 3 h, 2 workers per GPU
dalia/dalia.sh logs <job> -f                   # one line per trial, progress every 5 min
dalia/dalia.sh fetch                           # outputs/hpo/<study>/: trials.csv, best.json
```

`hpo.py` tunes on a validation set cut from the training configurations (below the density
cut); the above-cut holdout is scored once, on the best model, which is the only one registered
(`grl_trajectory_mo_deeponet_hpo_ood`) and turned into `submission_ood.csv`. Running the same
`--study` again resumes it. `--n-trials`, `--timeout-hours`, `--iterations`,
`--workers-per-gpu`, `--final-ensemble` and `--no-register` change the defaults.

Put `--dry-run` before any command to print what it would run on DALIA instead. Longer jobs:
`DALIA_SBATCH="--time=06:00:00" dalia/dalia.sh submit ...`.

Notes

- Only the `HOPSWORKS_*` lines of `.env` are copied to the remote folder (mode 600); the W&B key
  stays on your laptop.
- A new feature view (`--fv-version`) and its statistics are created by the first job that asks
  for them; run that one alone before a sweep over it. Use `--register` in sweeps sparingly:
  parallel registrations of one model name race for the version number.
- Nothing is logged to W&B from a training run: the CSV goes to the organisers first, and their
  score is transcribed afterwards with `log_official_score.py`.

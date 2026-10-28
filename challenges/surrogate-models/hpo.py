"""Hyperparameter search for MO_DeepOnet with Optuna; only the best model is published.

The search never looks at the holdout. The training configurations of the split (for
--split ood_density: below the density cut, minus the in_range holdout) are cut again into fit
and validation configurations; every trial trains on fit and is scored on validation, with
pallas_score (in-plasma mean R2 for the trajectory). TPE proposes the next trial from those
scores. The above-cut configurations are never seen by the search, as models/MO_DeepOnet.py
asks: a model tuned on them has learned the cut.

At the end the best trial's parameters are retrained on all the training configurations,
scored once on the holdout (in_range and above_cut), registered in the Hopsworks model registry
(unless --no-register) and turned into the submission CSV -- through train.publish, as any run.

Workers: one process per GPU slot (--workers-per-gpu on each visible GPU), sharing one study in
a journal file, so a search is resumed by running the same --study again. On DALIA,
dalia/hpo.sbatch runs it on 2 GPUs. Locally, from this folder:

    python hpo.py --iterations 300 --n-trials 4 --no-register --no-submission   # a smoke test

Everything lands in outputs/hpo/<study>/: journal.log (the study), trials.csv, best.json.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pallas_pipeline as pp
import pallas_score
import train
from train import log

MODEL_FILE = Path("models/MO_DeepOnet.py")
# The registered models/MO_DeepOnet.py PARAMS, tried first so the search starts from them
KNOWN_GOOD = {"width": 128, "branch_depth": 3, "trunk_depth": 3, "latent": 32,
              "activation": "tanh", "strategy": "split_branch", "coordinate_with": "none",
              "baseline": "none", "lr": 1e-3, "batch_size": 4096, "weight_decay": 0.0}


def suggest(trial):
    """The search space, as flat choices (the names trials.csv and best.json show)."""
    return {
        "width": trial.suggest_categorical("width", [64, 128, 256]),
        "branch_depth": trial.suggest_int("branch_depth", 2, 4),
        "trunk_depth": trial.suggest_int("trunk_depth", 2, 4),
        "latent": trial.suggest_categorical("latent", [16, 32, 64]),
        "activation": trial.suggest_categorical("activation", ["tanh", "gelu", "sin"]),
        "strategy": trial.suggest_categorical("strategy", ["split_branch", "split_both"]),
        "coordinate_with": trial.suggest_categorical("coordinate_with", ["none", "l_inj"]),
        "baseline": trial.suggest_categorical("baseline", ["none", "linear_z"]),
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4]),
    }


def model_params(choice, iterations, ensemble, seed):
    """MO_DeepOnet parameters for one point of the search space."""
    return {"branch_layers": [choice["width"]] * choice["branch_depth"],
            "trunk_layers": [choice["width"]] * choice["trunk_depth"],
            "latent": choice["latent"], "activation": choice["activation"],
            "strategy": choice["strategy"],
            "coordinate_with": [] if choice["coordinate_with"] == "none"
                               else [choice["coordinate_with"]],
            "baseline": None if choice["baseline"] == "none" else choice["baseline"],
            "lr": choice["lr"], "batch_size": choice["batch_size"],
            "weight_decay": choice["weight_decay"], "lr_decay": "cosine",
            "iterations": iterations, "ensemble": ensemble, "seed": seed}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    train.data_arguments(p, split="ood_density")
    p.add_argument("--study", help="study name: resumes it when it exists "
                                   "(default: deeponet_<split>_<date>)")
    p.add_argument("--n-trials", type=int, default=60, help="trials in the study, in total")
    p.add_argument("--timeout-hours", type=float, default=3.0,
                   help="no new trial starts after this; running ones finish")
    p.add_argument("--iterations", type=int, default=40000, help="per trial, and the final fit")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="training configurations held for the search's validation")
    p.add_argument("--workers-per-gpu", type=int, default=2)
    p.add_argument("--final-ensemble", type=int, default=1,
                   help="nets averaged in the published model (the search trains one)")
    p.add_argument("--no-register", action="store_true", help="do not upload the best model")
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)      # internal
    p.add_argument("--deadline", type=float, help=argparse.SUPPRESS)  # internal
    args = p.parse_args(argv)
    if args.study is None:
        args.study = f"deeponet_{args.split}_{time.strftime('%Y%m%d-%H%M')}"
    return train.finish_args(args)


def study_dir(args):
    return pp.HERE / "outputs" / "hpo" / args.study


def storage(args):
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    path = str(study_dir(args) / "journal.log")
    # an open-file lock rather than flock: safe on Lustre and NFS
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def search_split(data, fraction, seed):
    """Fit and validation configurations, both inside the split's training configurations."""
    configs = np.sort(data.train_rows["config"].unique())
    val = set(np.random.default_rng(seed + 1).choice(
        configs, size=int(round(fraction * len(configs))), replace=False).tolist())
    is_val = data.train_rows["config"].isin(val)
    take = lambda frame, mask: frame[mask.values].reset_index(drop=True)
    return {"X_fit": take(data.X_train, ~is_val), "fit_rows": take(data.train_rows, ~is_val),
            "X_val": take(data.X_train, is_val), "val_rows": take(data.train_rows, is_val),
            "n_fit": len(configs) - len(val), "n_val": len(val)}


# ---- worker: runs trials until the study is full or the deadline passes -----------------------

def run_worker(args):
    import optuna
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.chdir(pp.HERE)
    d = pd.read_pickle(study_dir(args) / "data.pkl")
    case = dataclasses.replace(pp.CASES[args.case], inputs=d["inputs"])
    study = optuna.load_study(study_name=args.study, storage=storage(args))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "") or "cpu"

    def objective(trial):
        params = model_params(suggest(trial), args.iterations, 1, args.seed)
        model, seconds = train.fit(case, MODEL_FILE, params, d["X_fit"], d["fit_rows"],
                                   args.target_scaling)
        scores = pallas_score.score_submission(
            pp.predictions(case, model, d["X_val"], d["val_rows"]),
            pp.answer_key(case, d["val_rows"]))
        trial.set_user_attr("train_seconds", round(seconds, 1))
        for t in case.targets:
            trial.set_user_attr(f"val_{t}", float(scores[t]))
        log(f"worker {args.worker} (GPU {gpu}) trial {trial.number}: validation mean R2 "
            f"{scores['mean']:.4f} in {seconds:.0f} s")
        return float(scores["mean"])        # NaN marks the trial failed

    done = (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED)
    study.optimize(objective, timeout=max(1.0, args.deadline - time.time()),
                   callbacks=[MaxTrialsCallback(args.n_trials, states=done)],
                   catch=(Exception,), gc_after_trial=True)


# ---- main: data, workers, then the best model --------------------------------------------------

def visible_gpus():
    listed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if listed is not None:
        return [g for g in listed.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return out.split()
    except (OSError, subprocess.CalledProcessError):
        return []


def main(argv=None):
    args = parse_args(argv)
    if args.worker is not None:
        return run_worker(args)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data = train.load_data(args)
    if data is None:                                # --rows export
        return None
    folder = study_dir(args)
    folder.mkdir(parents=True, exist_ok=True)
    split = search_split(data, args.val_fraction, args.seed)
    pd.to_pickle({"inputs": data.case.inputs, **split}, folder / "data.pkl")
    log(f"search: fit {split['n_fit']} configurations, validation {split['n_val']}, both inside "
        f"the training ones; the holdout ({', '.join(data.holdout_groups)}) is kept for the end")

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, constant_liar=True,
                                         n_startup_trials=10)
    study = optuna.create_study(study_name=args.study, storage=storage(args), sampler=sampler,
                                direction="maximize", load_if_exists=True)
    if not study.trials:
        study.enqueue_trial(KNOWN_GOOD)
    gpus = visible_gpus() or [""]                   # "" = CPU
    slots = [g for g in gpus for _ in range(args.workers_per_gpu)]
    deadline = time.time() + 3600 * args.timeout_hours
    log(f"study {args.study}: {args.n_trials} trials, {len(slots)} workers on "
        f"{'GPUs ' + ','.join(gpus) if gpus != [''] else 'CPU'}, "
        f"{args.iterations} iterations each, until {time.strftime('%H:%M', time.localtime(deadline))}")

    passthrough = [a for a in (argv if argv is not None else sys.argv[1:])
                   if a not in ("--no-register", "--no-submission")]
    workers = []
    for i, gpu in enumerate(slots):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        command = [sys.executable, "-u", str(pp.HERE / "hpo.py"), *passthrough,
                   "--study", args.study, "--worker", str(i), "--deadline", str(deadline)]
        workers.append(subprocess.Popen(command, env=env))
    reported = time.time()
    while any(w.poll() is None for w in workers):
        time.sleep(5)
        if time.time() - reported < 300:            # a progress line every 5 minutes
            continue
        reported = time.time()
        finished = [t for t in study.get_trials(deepcopy=False) if t.value is not None]
        if finished:
            log(f"{len(finished)} trials done, best validation mean R2 "
                f"{max(t.value for t in finished):.4f}")
    if any(w.returncode for w in workers):
        log(f"worker exit codes: {[w.returncode for w in workers]}")

    # The search's record
    trials = study.trials_dataframe()
    trials.to_csv(folder / "trials.csv", index=False)
    complete = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    if not complete:
        raise SystemExit("no trial finished: nothing to publish")
    best = study.best_trial
    try:
        importance = {k: round(v, 4) for k, v in
                      optuna.importance.get_param_importances(study).items()}
    except Exception as error:                      # needs a few finished trials
        importance = {"unavailable": str(error)}
    summary = {"study": args.study, "split": args.split, "trials": len(study.trials),
               "complete": len(complete), "best_trial": best.number,
               "best_validation_mean_r2": best.value, "best_choice": best.params,
               "best_validation": best.user_attrs, "importance": importance,
               "fit_configs": split["n_fit"], "validation_configs": split["n_val"]}
    (folder / "best.json").write_text(json.dumps(summary, indent=2))
    log(f"best: trial {best.number}, validation mean R2 {best.value:.4f}, {best.params}")
    log(f"importance: {importance}")
    shown = trials.sort_values("value", ascending=False).head(10)
    print(shown[[c for c in shown.columns if c in ("number", "value") or
                 c.startswith("params_")]].to_string(index=False), flush=True)

    # The best model: retrained on every training configuration, then published like any run
    module = pp.load_model_file(MODEL_FILE)
    params = {**module.PARAMS,
              **model_params(best.params, args.iterations, args.final_ensemble, args.seed)}
    log(f"retraining the best on {len(data.train_rows['config'].unique())} configurations: "
        f"{params}")
    model, seconds = train.fit(data.case, MODEL_FILE, params, data.X_train, data.train_rows,
                               args.target_scaling)
    log(f"trained in {seconds:.1f} s")
    model_name = f"{args.owner}_{args.case}_{MODEL_FILE.stem}_hpo".lower()
    if args.split == "ood_density":
        model_name += "_ood"
    hpo = {"study": args.study, "trial": best.number, "trials": len(study.trials),
           "validation_mean_r2": best.value, "choice": best.params}
    return train.publish(args, data, model, MODEL_FILE, params, seconds, model_name,
                         register=not args.no_register, extra_meta={"hpo": hpo},
                         extra_metrics={"hpo_validation_mean_r2": best.value},
                         note=f", best of {len(study.trials)} Optuna trials ({args.study})")


if __name__ == "__main__":
    main()

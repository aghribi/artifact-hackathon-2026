"""Train one model end to end without the notebook: the steps of surrogate_pipeline.ipynb, headless.

Feature view -> split -> fit -> holdout scores -> bundle -> model registry (optional) ->
submission CSV. It runs the same on a laptop and on a DALIA compute node, where
dalia/train.sbatch calls it. Examples, from this folder:

    python train.py --model-file models/torch_mlp.py
    python train.py --case trajectory --split ood_density --model-file models/MO_DeepOnet.py \
        --params '{"iterations": 80000, "ensemble": 4}' --register

Every run writes outputs/runs/<model name>_<run id>/ (result.json and the model bundle) and the
CSV in submissions/<model name>_v<registry version>/ (or _<run id> when not registered). The CSV
is the hand-in. Nothing goes to W&B from here: the organisers score the CSV first, and
log_official_score.py then transcribes their score.

hpo.py reuses load_data, fit and publish, so a searched model is split, scored and published
exactly like one trained here.
"""

import argparse
import dataclasses
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

import pallas_pipeline as pp
import pallas_score
from team_wandb import TIERS

# The tier each objective is ranked in (objectives.md); the OOD split is the Hard one.
DEFAULT_TIERS = {"trajectory": "medium", "direct_B": "easy", "inverse_B": "medium",
                 "inverse_A": "easy"}


def data_arguments(p, split="random"):
    """The options that choose the data, its split and where the results go (train and hpo)."""
    p.add_argument("--case", choices=sorted(pp.CASES), default="trajectory")
    p.add_argument("--inputs", nargs="+", help="a sublist of the case's inputs (default: all)")
    p.add_argument("--target-scaling", choices=("scorer", "none"), default="scorer")
    p.add_argument("--split", choices=pp.SPLITS, default=split)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--owner", default="grl", help="prefix of the feature view and model names")
    p.add_argument("--fv-version", type=int, default=1)
    p.add_argument("--tier", choices=TIERS,
                   help="the objective's tier, kept in result.json for log_official_score.py "
                        "(default: the case's)")
    p.add_argument("--no-submission", action="store_true", help="skip the test-file CSV")
    p.add_argument("--rows", choices=("hopsworks", "export", "cache"), default="hopsworks",
                   help="where the view's rows come from: the feature store; or export them to "
                        "data/fv_rows/ and stop (dalia.sh does this here); or read that export "
                        "(on DALIA, whose proxy blocks the Feature Query Service)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    data_arguments(p)
    p.add_argument("--model-file", help="e.g. models/torch_mlp.py")
    p.add_argument("--params", default="{}", help="JSON overrides of the model file's PARAMS")
    p.add_argument("--register", action="store_true", help="upload to the Hopsworks registry")
    args = p.parse_args(argv)
    if args.rows != "export" and not args.model_file:
        p.error("--model-file is required")
    args.params = json.loads(args.params)
    return finish_args(args)


def finish_args(args):
    if args.tier is None:
        args.tier = "hard" if args.split == "ood_density" else DEFAULT_TIERS[args.case]
    return args


def run_id():
    """A timestamp, plus the Slurm job (and array task) when there is one."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job = os.environ.get("SLURM_ARRAY_JOB_ID")
    if job:
        return f"{stamp}-j{job}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    job = os.environ.get("SLURM_JOB_ID")
    return f"{stamp}-j{job}" if job else stamp


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclasses.dataclass
class Data:
    """Everything a model is trained, scored and published with."""
    project: object
    fv: object
    case: pp.Case
    fv_name: str
    normalisation: dict
    td_version: int
    train_rows: pd.DataFrame
    holdout_rows: pd.DataFrame
    holdout_configs: list
    holdout_groups: dict
    X_train: pd.DataFrame
    X_holdout: pd.DataFrame
    test: pd.DataFrame

    @property
    def feature_columns(self):
        return list(self.X_train.columns)


def load_data(args):
    """Steps 1-3: feature view, rows, split and normalised inputs; None after --rows export."""
    os.chdir(pp.HERE)                               # outputs/ and submissions/ sit next to us
    case = pp.CASES[args.case]
    if args.inputs:
        case = dataclasses.replace(case, inputs=list(args.inputs))
    test = pd.read_parquet(pp.DATA_DIR / case.test_file)
    missing = [c for c in case.inputs if c not in test.columns]
    if missing:
        raise SystemExit(f"{missing} are not in {case.test_file}: the CSV could not be predicted")

    normalisation = case.default_normalisation()
    fv_name = f"{args.owner}_{args.case}_fv".lower()
    exported = pp.rows_file(fv_name, args.fv_version)
    if args.rows == "export" and exported.exists():
        log(f"{exported.relative_to(pp.HERE)} is already exported")
        return None
    if args.rows == "cache" and not exported.exists():
        raise SystemExit(f"{exported.relative_to(pp.HERE)} is missing: export it where the "
                         f"feature store is reachable (--rows export, which dalia.sh runs "
                         f"before syncing)")
    project = pp.connect(pp.PROJECT)
    fs = project.get_feature_store()
    fv = pp.get_feature_view(fs, fv_name, args.fv_version)
    if fv is None:
        fv = pp.create_feature_view(fs, case, fv_name, args.fv_version, normalisation,
                                    description=f"{args.case}: {', '.join(case.inputs)} -> "
                                                f"{', '.join(case.targets)}")
        log(f"created {fv_name} v{args.fv_version}")
    else:
        log(f"using the existing {fv_name} v{args.fv_version}")
    td_version = pp.prepare_statistics(fv)          # an export makes sure the statistics exist
    if args.rows == "export":
        exported.parent.mkdir(parents=True, exist_ok=True)
        rows = fv.query.read()
        rows.to_parquet(exported)
        log(f"exported {len(rows):,} rows to {exported.relative_to(pp.HERE)}")
        return None
    rows = pp.read_rows(fv, case, exported if args.rows == "cache" else None)
    train_configs, holdout_configs, holdout_groups = pp.split_configs(
        rows, case, args.split, args.holdout_fraction, args.seed)
    train_rows = rows[rows.config.isin(train_configs)].reset_index(drop=True)
    holdout_rows = rows[rows.config.isin(holdout_configs)].reset_index(drop=True)
    log(f"train {len(train_configs)} configurations ({len(train_rows):,} rows), held out "
        + ", ".join(f"{g} {len(m)}" for g, m in holdout_groups.items()))
    return Data(project=project, fv=fv, case=case, fv_name=fv_name,
                normalisation=normalisation, td_version=td_version, train_rows=train_rows,
                holdout_rows=holdout_rows, holdout_configs=holdout_configs,
                holdout_groups=holdout_groups, X_train=pp.transform(fv, train_rows, case),
                X_holdout=pp.transform(fv, holdout_rows, case), test=test)


def fit(case, model_file, params, X, rows, target_scaling):
    """Step 4: (fitted SurrogateModel, seconds)."""
    module = pp.load_model_file(model_file)
    started = time.time()
    model = pp.SurrogateModel(module.build_model(**params), case.targets, target_scaling)
    model.fit(X, rows)
    return model, time.time() - started


def publish(args, data, model, model_file, params, train_seconds, model_name, register,
            extra_meta=None, extra_metrics=None, note=""):
    """Steps 5-8: holdout scores, bundle, registry (when `register`), submission CSV, result.json."""
    rid = run_id()
    case = data.case
    holdout_pred = pp.predictions(case, model, data.X_holdout, data.holdout_rows)
    holdout_key = pp.answer_key(case, data.holdout_rows)
    in_group = lambda frame, members: frame[frame.config.isin(members)].reset_index(drop=True)
    group_scores = {group: pallas_score.score_submission(in_group(holdout_pred, members),
                                                         in_group(holdout_key, members))
                    for group, members in data.holdout_groups.items()}
    scores = group_scores.get("above_cut", group_scores.get("holdout"))
    for group, s in group_scores.items():
        log(f"{group}: mean R2 {s['mean']:.4f}  "
            + "  ".join(f"{t} {s[t]:.4f}" for t in case.targets))
    if "in_range" in group_scores:
        log(f"gap (in range - above cut): {group_scores['in_range']['mean'] - scores['mean']:.4f}")

    run_dir = Path("outputs/runs") / f"{model_name}_{rid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if case.positional:
        pp.r2_by_position(case, holdout_pred, holdout_key).to_csv(
            run_dir / "r2_by_position.csv", index=False)

    model_file = Path(model_file)
    description = (f"{args.case}: {model_file.stem} {params}, targets on the "
                   f"{args.target_scaling} scale, {data.fv_name} v{args.fv_version}, "
                   f"{args.split} holdout (seed {args.seed}), run {rid}{note}")
    meta = {"case": args.case, "inputs": case.inputs, "targets": case.targets,
            "feature_columns": data.feature_columns, "normalisation": data.normalisation,
            "feature_view": data.fv_name, "feature_view_version": args.fv_version,
            "td_version": data.td_version, "model_file": model_file.stem, "params": params,
            "target_scaling": args.target_scaling, "split": args.split, "seed": args.seed,
            "holdout_configs": data.holdout_configs, "holdout_groups": data.holdout_groups,
            "holdout_scores": {group: {k: v for k, v in s.items() if isinstance(v, float)}
                               for group, s in group_scores.items()},
            "run_id": rid, **(extra_meta or {})}
    metrics = {"holdout_mean_r2": scores["mean"],
               **{f"holdout_r2_{pp.fs_name(t)}": scores[t] for t in case.targets}}
    if "in_range" in group_scores:
        metrics["holdout_in_range_mean_r2"] = group_scores["in_range"]["mean"]
        metrics["holdout_gap"] = group_scores["in_range"]["mean"] - scores["mean"]
    metrics.update(extra_metrics or {})
    bundle = pp.save_bundle(model, model_file, meta, run_dir / "model")
    version = None
    if register:
        version = pp.register_model(data.project, bundle, model_name, data.fv, data.td_version,
                                    metrics, description).version
        log(f"registered {model_name} v{version}")

    # Submission, from the model in memory (the notebook reloads the same one from the registry)
    submission_path = None
    if not args.no_submission:
        X_test = pp.transform(data.fv, data.test, case)[data.feature_columns]
        submission = pp.predictions(case, model, X_test, data.test)
        folder = f"{model_name}_v{version}" if version else f"{model_name}_{rid}"
        submission_path = pp.write_submission(
            case, submission, data.test,
            Path("submissions") / folder / pp.submission_file(case, args.split))
        log(f"wrote {submission_path} ({len(submission):,} rows)")

    result = {"run_id": rid, "model_name": model_name, "registry_version": version,
              "case": args.case, "split": args.split, "tier": args.tier,
              "model_file": str(model_file), "params": params, "seed": args.seed,
              "train_seconds": train_seconds, "metrics": metrics,
              "group_scores": meta["holdout_scores"],
              "submission": str(submission_path) if submission_path else None,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), **(extra_meta or {})}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    if submission_path:
        shutil.copy(submission_path, run_dir / submission_path.name)
    log(f"done: {run_dir}")
    return result


def main(argv=None):
    args = parse_args(argv)
    data = load_data(args)
    if data is None:                                # --rows export
        return None
    model_file = Path(args.model_file)
    module = pp.load_model_file(model_file)
    params = {**getattr(module, "PARAMS", {}), **args.params}
    log(f"training {model_file.stem} with {params}")
    model, train_seconds = fit(data.case, model_file, params, data.X_train, data.train_rows,
                               args.target_scaling)
    log(f"trained in {train_seconds:.1f} s")
    model_name = f"{args.owner}_{args.case}_{model_file.stem}".lower()
    if args.split == "ood_density":
        model_name += "_ood"
    return publish(args, data, model, model_file, params, train_seconds, model_name,
                   args.register)


if __name__ == "__main__":
    main()

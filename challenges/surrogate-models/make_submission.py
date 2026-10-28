"""Build a hand-in folder: one submission CSV per case, each from the model chosen for it.

    python make_submission.py                       # into submissions/best_<today>/
    python make_submission.py --out submissions/final

Every model is loaded from the team's Hopsworks model registry (PROJECT of pallas_pipeline) with
its own bundled code and asked to predict the case's test file; the CSV is checked and written
under the name the organisers expect. Models from surrogate_pipeline.ipynb / train.py go through
their feature view's normalisations; the moments models (their own notebook) predict straight
from the test file's columns. manifest.json beside the CSVs records, per file, the model, its
registered scores and the CSV's md5. Edit CHOSEN when a better model is registered; a case left
out of it is not handed in.
"""

import argparse
import dataclasses
import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import pallas_pipeline as pp

# case -> (registered model, version), all in the team's project. "ood" is the trajectory case
# with split "ood_density" (submission_ood.csv). Inverse B is not handed in.
CHOSEN = {
    "inverse_A": ("grl_inverse_a_mlp_sklearn", 1),
    "direct_B": ("grl_direct_b_mlp_sklearn", 1),
    "trajectory": ("sj_trajectory_bayesian_nn", 5),
    "ood": ("sj_trajectory_bayesian_nn_ood", 1),
    "moments": ("m_moments_hybrid_film_avg_centroids_all", 1),
}

# The Medium moments objective (objectives.md); not a case of the notebook.
MOMENTS = SimpleNamespace(
    test_file="test_moments_planes.parquet", submission_file="submission_moments.csv",
    ids=["config", "plane_mm"],
    targets=[f"{kind}_{c}" for kind in ("mean", "sigma")
             for c in ("zeta", "y", "z", "ux", "uy", "uz")] + ["q_tot_pC"])


def predict_case(fs, model, meta):
    """(case, test inputs, submission frame) for one loaded model."""
    if meta["case"] == "moments":
        test = pd.read_parquet(pp.DATA_DIR / MOMENTS.test_file)
        out = model.predict(test).reset_index(drop=True)
        submission = pd.concat([test[MOMENTS.ids].reset_index(drop=True),
                                out[MOMENTS.targets]], axis=1)
        return MOMENTS, test, submission
    case = dataclasses.replace(pp.CASES[meta["case"]], inputs=meta["inputs"])
    fv = fs.get_feature_view(meta["feature_view"], meta["feature_view_version"])
    if fv.transformation_functions:
        fv.init_batch_scoring(training_dataset_version=meta["td_version"])
    test = pd.read_parquet(pp.DATA_DIR / case.test_file)
    X = pp.transform(fv, test, case)[meta["feature_columns"]]
    return case, test, pp.predictions(case, model, X, test)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("submissions") / f"best_{date.today()}")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    project = pp.connect()
    fs = project.get_feature_store()
    registry = project.get_model_registry()
    manifest = {"project": project.name, "built": str(date.today()), "files": {}}
    for task, (name, version) in CHOSEN.items():
        model, meta = pp.load_registered_model(project, name, version, "outputs/downloaded",
                                               own_code=True)
        case, test, submission = predict_case(fs, model, meta)
        if not np.isfinite(submission[case.targets].to_numpy()).all():
            raise ValueError(f"{name} v{version} predicts non-finite values")
        filename = (MOMENTS.submission_file if case is MOMENTS
                    else pp.submission_file(case, meta.get("split")))
        path = pp.write_submission(case, submission, test, args.out / filename)
        entry = registry.get_model(name, version=version)
        manifest["files"][filename] = {
            "task": task, "model": name, "version": version, "split": meta.get("split"),
            "registered_metrics": entry.training_metrics, "description": entry.description,
            "rows": len(submission), "md5": hashlib.md5(path.read_bytes()).hexdigest()}
        print(f"{path}  <-  {name} v{version}  ({len(submission):,} rows)", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()

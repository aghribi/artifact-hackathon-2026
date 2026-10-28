"""Check that this machine can run surrogate_pipeline.ipynb: packages, credentials, Hopsworks, W&B.

Run from this folder with the environment's own Python:
    .venv/bin/python check_setup.py

Credentials come from the environment or the nearest .env in this folder or above it
(copy .env.example to the repository root as .env and fill it in). Nothing is written anywhere.
"""

import os
import sys
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = "m4cast_pallas"            # the full public pack, as in pallas_pipeline.PROJECT
failures = []


def report(ok, what, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {what}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(what)
    return ok


def pinned():
    """{package: version} from requirements.txt."""
    pins = {}
    for line in (HERE / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==")
            pins[name.split("[")[0]] = version
    return pins


def check_packages():
    report(sys.version_info[:2] == (3, 11), f"Python {sys.version.split()[0]}",
           "" if sys.version_info[:2] == (3, 11) else "the pins are checked on 3.11")
    for name, version in pinned().items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            report(False, f"{name} installed", "missing: uv pip install -r requirements.txt")
            continue
        report(installed == version, f"{name}=={installed}",
               "" if installed == version else f"requirements.txt pins {version}")


def check_credentials():
    sys.path.insert(0, str(HERE))
    from team_wandb import load_env
    found = load_env()
    report(found is not None, "a .env file", str(found) if found else "copy .env.example to .env")
    for name in ("HOPSWORKS_HOST", "HOPSWORKS_API_KEY", "WANDB_API_KEY"):
        report(bool(os.environ.get(name)), f"{name} set")


def check_hopsworks():
    if not (os.environ.get("HOPSWORKS_HOST") and os.environ.get("HOPSWORKS_API_KEY")):
        return report(False, "Hopsworks", "no credentials")
    import requests
    host = os.environ["HOPSWORKS_HOST"]
    headers = {"Authorization": f"ApiKey {os.environ['HOPSWORKS_API_KEY']}"}
    try:
        answer = requests.get(f"https://{host}/hopsworks-api/api/project", headers=headers,
                              timeout=30)
    except requests.RequestException as error:
        return report(False, f"reach {host}", str(error)[:120])
    if not report(answer.ok, f"reach {host} with the API key", f"HTTP {answer.status_code}"):
        return False
    projects = sorted(p.get("project", p)["name"] for p in answer.json())
    if not report(PROJECT in projects, f"the key opens {PROJECT}",
                  "" if PROJECT in projects else f"it opens {projects}; ask for access"):
        return False

    import hopsworks
    project = hopsworks.login(host=host, project=PROJECT,
                              api_key_value=os.environ["HOPSWORKS_API_KEY"])
    rows = len(project.get_feature_store().get_feature_group("pallas_campaign_b", version=1).read())
    report(rows == 5286, f"read pallas_campaign_b from {PROJECT}", f"{rows:,} rows")
    report(project.get_model_registry() is not None, "open the model registry")


def check_wandb():
    if not os.environ.get("WANDB_API_KEY"):
        return report(False, "W&B", "no WANDB_API_KEY")
    import wandb
    try:
        viewer = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30).viewer
        report("artifact-hackathon-2026" in (viewer.teams or []), "W&B key in the hackathon team")
    except Exception as error:
        report(False, "W&B key", str(error)[:120])


def check_data():
    pack = Path(os.environ.get("PALLAS_DATA_DIR", HERE / "data" / "pallas_hackathon_data"))
    report((pack / "test_trajectory.parquet").is_file(), "local data pack (test inputs)", str(pack))


if __name__ == "__main__":
    for check in (check_packages, check_credentials, check_hopsworks, check_wandb, check_data):
        print(f"-- {check.__name__.removeprefix('check_')}")
        try:
            check()
        except Exception as error:
            report(False, check.__name__, f"{type(error).__name__}: {str(error)[:160]}")
    print("\nall good" if not failures else f"\n{len(failures)} problem(s): {', '.join(failures)}")
    sys.exit(1 if failures else 0)

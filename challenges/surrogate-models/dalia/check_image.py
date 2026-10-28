"""The versions train.py depends on and the GPU it sees: run inside the image after a build."""

from importlib import metadata

import torch

for package in ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "joblib", "torch",
                "deepxde", "optuna", "wandb", "hopsworks"]:
    try:
        print(f"{package:13s} {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package:13s} MISSING")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU           {gpu.name}, {gpu.total_memory / 1e9:.0f} GB")
else:
    print("GPU           none visible (expected after `build`, which runs without one; "
          "`dalia.sh check` runs this on a GPU)")

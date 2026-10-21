"""Train the joint native-waveform cycle AE from the terminal, without Jupyter.

Model and training definitions are read from mbl_autoencoder_training.ipynb.
Only function/class definitions are loaded; notebook execution and display cells
are never run. This keeps terminal and notebook training on the same code.

Example:
    python -u train_mbl_cycle_autoencoder.py --device cuda --epochs 160
"""

import argparse
import ast
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent


def load_definitions(output, device):
    """Load reusable definitions with a headless plotting backend."""
    import copy
    import hashlib
    import time

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F

    notebook_path = ROOT / "mbl_autoencoder_training.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = dict(
        __name__=__name__, Path=Path, ROOT=ROOT, OUTPUT=output,
        copy=copy, hashlib=hashlib, json=json, time=time,
        plt=plt, np=np, pd=pd, torch=torch, nn=nn, F=F,
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if definitions:
            module = ast.Module(body=definitions, type_ignores=[])
            exec(compile(module, f"{notebook_path}:cell{index}", "exec"), namespace)
    required = {"CycleData", "train_cycle_model", "load_cycle_model", "evaluate_cycle_model"}
    if not required.issubset(namespace):
        raise RuntimeError(f"Missing notebook definitions: {sorted(required - namespace.keys())}")
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on a GPU with CUDA-enabled PyTorch, or use --device cpu.")
    return namespace, resolved_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "mbl_dwn0_cmp1_normal_without_official_holdout.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "mbl_cycle_autoencoder")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="Number of whole cycles per batch (default: 4)")
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0 (default: auto)")
    parser.add_argument("--threads", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Save cycle partitions and normalization without training")
    mode.add_argument("--evaluate-only", action="store_true", help="Evaluate the saved cycle model without training")
    args = parser.parse_args(argv)
    for name in ["epochs", "patience", "batch_size", "latent", "threads"]:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.lr < float("inf"):
        parser.error("--lr must be finite and positive")
    args.input = str(args.input.resolve())
    args.output = str(args.output.resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if not Path(args.input).is_file():
        raise FileNotFoundError(args.input)
    if args.evaluate_only and not (output / "cycle_ae.pt").is_file():
        raise FileNotFoundError(output / "cycle_ae.pt")
    definitions, args.device = load_definitions(output, args.device)
    definitions["torch"].set_num_threads(args.threads)
    print(f"Device: {args.device}", flush=True)
    print(f"Input: {args.input}", flush=True)
    print(f"Output: {output}", flush=True)
    print(f"Batch size: {args.batch_size} whole cycles", flush=True)
    data = definitions["CycleData"](args)
    if args.prepare_only:
        data.save_preprocessing()
        print("Saved cycle partitions and training-only normalization; no training performed.", flush=True)
        return
    if args.evaluate_only:
        model, _ = definitions["load_cycle_model"](output, args.device, data.config["signature"])
    else:
        model = definitions["train_cycle_model"](data)
    _, _, metrics = definitions["evaluate_cycle_model"](data, model)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved cycle decisions and PV reconstruction errors to {output}", flush=True)


if __name__ == "__main__":
    main()

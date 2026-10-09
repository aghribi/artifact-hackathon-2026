# Demos for "AI for Science: putting physics into learning"

Three small CPU demos, about 1 minute each, one per part of the lecture.

| Notebook | Part | What it shows |
|---|---|---|
| `demo1_pinn_oscillator.ipynb` | I. Soft constraints | Data-only network vs PINN on a damped oscillator observed only at the start; PINN with wrong physics |
| `demo2_symmetry_cluster.ipynb` | II. Hard constraints | Raw coordinates vs augmentation vs canonicalisation vs invariant descriptors, on cluster energies |
| `demo3_diffusion_doublewell.ipynb` | III. Probabilistic models | A diffusion model learns a 2D Boltzmann distribution; free-energy profile check |

## Run

```bash
uv sync                 # numpy, matplotlib, CPU torch, jupyter
uv run jupyter lab
```

Figures are written to `figures/` (they are also embedded in the slides).

`fem_heat.py` (not a notebook) makes the finite-element figure of the "problem" slide in Part I:
a deliberately rough P1 solution of the 1D heat equation with scikit-fem
(`uv run python fem_heat.py`).

The `.py` files are the sources; regenerate the notebooks with
`uv run python py2nb.py demo*.py`.

# Accelerators and AI

**Speaker:** Adnan Ghribi
**Slot:** Monday, October 12 — 11:05–12:05 (50 min talk + 10 min Q&A)

## Abstract

A 60-minute tour of AI applied to the accelerator itself — from the physics roots of modern AI (McCulloch-Pitts to the 2024 Nobel Prize) through 30 years of AI in accelerator operations, to three hands-on use cases with live notebooks: RF fault detection (autoencoders), beam tuning (Bayesian optimisation with Cheetah), and surrogate models (MLP + uncertainty quantification). Each use case is a direct methodological preview of one of this week's three challenges, and the lecture closes by handing off to the case holders' briefings at 14:50.

## Quick links

| | |
|---|---|
| **Notebook 01** — RF fault detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aghribi/artifact-hackathon-2026/blob/main/lectures/02-accelerators-and-ai/notebooks/01_fault_detection/notebook.ipynb) → preview of **Anomaly Detection** (ESS) |
| **Notebook 02** — Beam tuning with Cheetah + BO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aghribi/artifact-hackathon-2026/blob/main/lectures/02-accelerators-and-ai/notebooks/02_beam_tuning/notebook.ipynb) → preview of **Optimisation** (CLEAR/CLARA) |
| **Notebook 03** — Neural network surrogate | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aghribi/artifact-hackathon-2026/blob/main/lectures/02-accelerators-and-ai/notebooks/03_surrogate_model/notebook.ipynb) → preview of **Surrogate Models** (PALLAS) |

Live slides URL: _TBD — to be published via GitHub Pages closer to the event._

## Lecture structure (50 min + 10 min Q&A)

| Time | Topic |
|------|-------|
| 0–10 min | Introduction — what AI is/isn't, brief history (1943→2024 Nobel), AI through the physicist lens |
| 10–14 min | 30 years of AI at accelerators + 2026 technology-maturity snapshot |
| 14–18 min | Practical foundations — the ML pipeline |
| 18–27 min | **Use case 1**: RF fault detection — autoencoders on IQ waveforms |
| 27–36 min | **Use case 2**: Beam tuning — Bayesian optimisation with Cheetah |
| 36–43 min | **Use case 3**: Surrogate models — replacing tracking codes |
| 43–50 min | Future directions & open problems, summary |
| 50–60 min | Q&A → hands off to the 14:50 challenge briefings |

## Run locally

```bash
git clone https://github.com/aghribi/artifact-hackathon-2026.git
cd artifact-hackathon-2026/lectures/02-accelerators-and-ai

# Create environment (conda)
conda env create -f environment.yml
conda activate aissai-lecture-02

# Render slides
quarto render index.qmd

# Launch notebooks
jupyter lab
```

## Repository structure

```
├── index.qmd                            ← Quarto reveal.js lecture slides
├── custom.scss                          ← Visual theme (ARTIFACT palette)
├── _quarto.yml                          ← Quarto project config
├── environment.yml                      ← Conda environment
├── requirements.txt                     ← pip requirements (for Colab)
│
├── notebooks/
│   ├── 01_fault_detection/
│   │   ├── notebook.ipynb              ← Autoencoder on synthetic RF waveforms
│   │   └── generate_data.py            ← Synthetic IQ waveform generator
│   ├── 02_beam_tuning/
│   │   └── notebook.ipynb              ← BO with Cheetah (proton lattice)
│   └── 03_surrogate_model/
│       └── notebook.ipynb              ← MLP surrogate + ensemble UQ
│
├── notebooks_rendered/                  ← Static HTML previews embedded in the slides
└── assets/                              ← Images used by the slides
```

## Key tools

| Tool | Role |
|------|------|
| [Cheetah](https://github.com/desy-ml/cheetah) | PyTorch-based differentiable beam dynamics — simulation backbone for notebooks 02 & 03 |
| [scikit-optimize](https://scikit-optimize.github.io/) | Gaussian Process Bayesian optimisation |
| [Quarto](https://quarto.org/) | Reproducible reveal.js slides from Markdown + Python |

## Key references

- Kaiser et al., *Phys. Rev. Accel. Beams* **27**, 054601 (2024) — Cheetah simulator
- Tennant et al., *PRAB* **23**, 114601 (2020) — SRF fault classification at JLab
- Duris et al., *PRL* **124**, 124801 (2020) — Bayesian optimisation at LCLS
- Ghribi et al., *Europhysics News* **56**(1), 15–19 (2025) — ARTIFACT / KARA −30% result
- AccML living review: https://aghribi.github.io/acc-ml-living-review

## Additional resources

- Ghribi, *ARTIFACT: From AI History to Federated Infrastructures* — GANIL Physics Seminar, March 2026 (state-of-the-art material adapted for slides on AI's physics lineage and the 2026 technology-maturity snapshot)

## License

Code: MIT · Slides content: CC BY 4.0

# Current optimization results

The current result is the **adaptive single-shot, model-assisted BO/RCDS run**: 100 Cheetah episodes per method, seeds 42–141, centered (1, 1) mm targets, 10,000 particles, live episode/shot stochasticity, 0.1% supply ripple, powered VBC and L4H as a drift. It supersedes the 192-shot fixed-budget run as the default result.

| Method | Mean shots | Median | 10th–90th percentile | Range | Mean score | Median score |
|---|---:|---:|---|---|---:|---:|
| BO | 5.64 | 6 | 2–9 | 1–13 | 0.430 | 0.382 |
| RCDS | 6.20 | 6 | 2–11 | 1–15 | 0.500 | 0.433 |

Shots exclude the reset reading and include every measured correction and restoration. Both methods completed all 100 branches with beam on the IP screen. There were **1,184 total steps**, **200 reset readings** and **zero screen changes**. A completed branch does not mean every final size/position tolerance was satisfied.

## Compute timings

| Method | Mean ± SD | Median | 10th–90th percentile |
|---|---|---|---|
| BO | 4.14 ± 1.39 s | 4.56 s | 2.36–5.56 s |
| RCDS | 4.01 ± 1.37 s | 4.44 s | 2.27–5.45 s |

These are **local simulation/model/optimizer compute times**, not accelerator operating times. The per-method timer includes environment construction/reset, any threading, nominal model-assisted proposals, residual search and centering; shared nominal response setup is outside that timer. The complete batch, including shared setup and JSON export, took **177.78 s (2.96 min)** with eight episodes in parallel. Worker times must not be summed to infer batch wall time. No hardware settling or acquisition delay is inserted.

## Stopping and quality

The focus uses the MBRL-style aggregate size/loss tolerance of 0.15 and patience of three significant non-improvements. BO additionally stops when GP expected improvement is below the marginal benchmark shot cost. Model proposals and inverse/observed-response centering avoid spending measurements on model computations; there are no unconditional three-shot repeats or replay acquisitions.

Relative to the completed fixed-budget branches, mean shot counts fell by **97.1% for BO** and **96.8% for RCDS**. On matched seeds, mean physical mismatch (excluding shot cost) changed from **0.431 to 0.356** for BO and **0.356 to 0.415** for RCDS. The earlier run averaged three final readings, while this run scores the last actual single shot; the quality comparison is descriptive. RCDS trades some residual quality for fewer shots.

Strict final per-plane checks passed in 30 BO and 26 RCDS episodes; the benchmark relative-error hit rates within 10% / 25% / 50% were 51% / 87% / 99% for BO and 44% / 77% / 97% for RCDS. Full definitions are in [ADAPTIVE_VALIDATION.md](ADAPTIVE_VALIDATION.md).

## Data and figures

- [Current compact results](results/latest_results.json): aggregate statistics, timing definitions and all per-episode metrics; sufficient to regenerate the histograms without tracking intermediates.
- [Completed audit](results/audit.json) and [algorithm source checksums](results/algorithm_source_sha256.json). Raw acquired data, intermediate artifacts and superseded runs have been removed.
- Talk histograms: [steps PNG](../talk_figures/model_svd_opt/model_svd_opt_100_episodes_steps.png), [steps PDF](../talk_figures/model_svd_opt/model_svd_opt_100_episodes_steps.pdf), [scores PNG](../talk_figures/model_svd_opt/model_svd_opt_100_episodes_score.png), [scores PDF](../talk_figures/model_svd_opt/model_svd_opt_100_episodes_score.pdf).
- [Combined figure](../talk_figures/model_svd_opt/model_svd_opt_100_episodes_all.png) and [talk data JSON](../talk_figures/model_svd_opt/model_svd_opt_100_episodes.json).

```sh
.venv/bin/python talk_figures/scripts/model_svd_episode_stats.py
```

## Retention

Raw episode data and historical run folders have been removed. The compact current result preserves episode scores, shot counts and timings for reproducing the talk histograms; earlier comparisons above are retained as summary statistics only.

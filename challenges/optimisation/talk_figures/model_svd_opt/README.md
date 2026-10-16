# Adaptive model-SVD optimization: 100 episodes

These figures show the current model-assisted BO and RCDS results from `model-svd_opt/results/adaptive_100_episodes/`. Each method has 100 Cheetah episodes (seeds 42–141), with centered 1 mm × 1 mm targets, 10,000 particles, episode and shot stochasticity, powered VBC and L4H replaced by a drift.

| Method | Mean shots | Median shots | Mean score | Median score | Mean compute time |
|---|---:|---:|---:|---:|---:|
| BO | 5.64 | 6 | 0.430 | 0.382 | 4.14 s |
| RCDS | 6.20 | 6 | 0.500 | 0.433 | 4.01 s |

- [Episode steps histogram](model_svd_opt_100_episodes_steps.png) ([PDF](model_svd_opt_100_episodes_steps.pdf)). Each step is one measured shot; reset readings are excluded. All correction and restoration shots are counted.
- [Final score histogram](model_svd_opt_100_episodes_score.png) ([PDF](model_svd_opt_100_episodes_score.pdf)). This is the last actual single-shot `SimEnv` benchmark score, including mismatch, loss and shot-count penalty; lower is better.
- [Combined figure](model_svd_opt_100_episodes_all.png) ([PDF](model_svd_opt_100_episodes_all.pdf)). Solid lines mark means; dashed lines mark medians. Both methods use identical histogram bins.
- [Compact data](model_svd_opt_100_episodes.json) includes every episode, aggregate statistics, source checksum and timing/count definitions.

There were 1,184 steps plus 200 reset readings, zero screen changes and no failed branches. All branches ended with beam at the IP, but strict final target checks passed in 30 BO and 26 RCDS episodes. Completion does not imply convergence to every tolerance.

Times describe local simulation/model/optimizer computation, with no machine settling or acquisition delay. The batch took 177.78 s with eight workers. Shared nominal response setup is outside the per-branch timer. See [current results](../../model-svd_opt/RESULTS.md) for timing scope, stopping criteria and quality comparisons. The older 192-shot run is historical.

Regenerate from the repository root:

```sh
.venv/bin/python talk_figures/scripts/model_svd_episode_stats.py
```

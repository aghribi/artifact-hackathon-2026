# Challenge Objectives — Optimisation (CLEAR / CLARA)

**⚠️ Organiser draft — not reviewed by the case owners (Amelia Pollard, Antonio Gilardi).**
Grounded in real CLARA data, but expect corrections. CLEAR isn't covered — its data
situation is still unexplored; if you're working CLEAR, the *methods* below still apply,
but you'll need CLEAR's own equivalent of the real captures used here.

Objectives only — see [`scientific_case.md`](scientific_case.md) (still TBD) and
[`GRADING.md`](../../GRADING.md) for how scoring and W&B logging work across all three
challenges.

**The subject of this challenge is the optimisation method, not the surrogate.**
`kickoff_notebook.ipynb` builds one real-data surrogate (settings → real CLARA spot size)
once, then runs six different optimisation methods against the same objective so they're
directly comparable. **Score for every tier:** best objective value found (mm, lower is
better) *and* evaluations needed to reach it — report both, since the cheapest method to a
good-enough answer is usually more valuable than the single best number on a real machine,
where every evaluation costs machine time. Log both to the shared W&B project (see
`GRADING.md`) as `optim/best_mm` and `optim/n_evals` each run.

## 🟢 Easy — classical methods

Run `kickoff_notebook.ipynb` to the end: four classical methods (grid search, Nelder-Mead
simplex, Powell, gradient-based via autodiff through the surrogate) against the same 2-knob
real objective.

- **Bar:** reference run, all four converge to **6.366mm** at the edge of the explored
  range; evaluations needed: gradient-based 6, Nelder-Mead 8, grid search 81, Powell 128
  (reproduce with the notebook — no separate `score.py` yet, the notebook *is* the
  reference run).
- Try one of the cited-but-not-run classical methods (random search, simulated annealing,
  a genetic algorithm) yourself and add it to the comparison table.

## 🟡 Medium — Bayesian Optimisation

Beat or match the Easy bar with Bayesian Optimisation (notebook section 6), in fewer
evaluations than grid search needed.

- **Bar:** reference BO run, **6.366mm in 40 evaluations** (vs. grid search's 81).
- Use the full raw archive instead of the 1,370-capture sample (CC-IN2P3 account required
  — see the [access guide](../../access-guide.md); data at
  `/sps/m4cast/artifact_hackathon_2026/optimisation/extracted/`, 2.4TB, ~20 camera
  stations) to train a better surrogate, then re-run all methods against it.
- Pick a different pair of magnets (or a 3rd dimension) and report whether BO's advantage
  over grid search holds up.

## 🔴 Hard — beyond BO

- Get reinforcement learning (notebook section 6) to *beat* BO, not just run — the
  reference RL run (REINFORCE, 80 episodes) lands at 6.641mm, clearly behind BO's 6.366mm
  in half the evaluations. Better policy, better reward shaping, a different RL algorithm
  (CMA-ES as a derivative-free alternative is a reasonable stand-in if pure RL stays
  uncompetitive) are all fair game — report what changed and why it helped.
- Scale past 2 magnets to most/all of the real live settings in the sample (20 in the
  provided sample, more in the full archive). Classical grid search becomes intractable
  (`N^D`); this is where TuRBO-style trust-region BO (cited in the notebook) earns its
  keep — implement it, or show concretely why vanilla GP-BO degrades as dimensions grow.
- Once a real CLARA Cheetah lattice exists (currently a placeholder), compare this
  learned surrogate against physics-based predictions, and build the hybrid digital-twin
  surrogate described in the notebook's section 6.

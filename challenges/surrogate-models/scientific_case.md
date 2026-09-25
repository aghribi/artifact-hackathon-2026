# Scientific Case — Surrogate Models (PALLAS)

## Background

A laser-plasma accelerator replaces the metres of radio-frequency cavity in a conventional
machine with a few millimetres of plasma. A high-power laser pulse is focused into a gas cell;
it drives a plasma wave in its own wake, and the electric field inside that wave is three
orders of magnitude stronger than anything a metal cavity can hold. Electrons injected into the
wave ride it and leave the cell at tens to hundreds of MeV.

PALLAS, at IJCLab (Orsay), is a laser-plasma injector of this kind. Its gas cell holds a
helium/nitrogen mixture: the helium supplies the plasma, and electrons stripped from the inner
shell of the nitrogen are released deep inside the wave, where they are trapped and accelerated.
The operator controls a small number of knobs — the pressure in the injection region, the
nitrogen concentration, the focal position of the laser, the laser amplitude, and in the second
campaign the whole shape of the gas density profile along the cell — and gets one electron bunch
out, characterised by its energy, energy spread, charge, emittance and divergence.

The data in this challenge are particle-in-cell (PIC) simulations of that machine, run with
Smilei in its cylindrical envelope mode. A PIC simulation resolves the plasma wave and the
individual macro-particles inside it, and it is the reason the map from knobs to bunch is known
exactly for every sample here: there is no diagnostic uncertainty, no shot-to-shot jitter and no
calibration. One simulation costs on the order of hours on a compute cluster.

## Motivation

That cost is the problem. Everything an operator or a designer wants to do with this machine —
scan a parameter, optimise a working point, understand which knob caused a change, fit a
measurement back to the settings that produced it, control the machine in a feedback loop —
needs the answer in milliseconds, not hours. A surrogate model is the standard answer: fit a
fast model to a corpus of simulations and query it instead.

The question a hackathon can settle, and a single accuracy number cannot, is **what such a
model actually gives up.** Three things are worth knowing before anyone puts a surrogate online:

- **The cost of dropping the physics.** A network that knows nothing about plasmas can fit this
  map very well inside the sampled region. A model built on the physics should be worse at the
  fit and, in principle, better where the data runs out — but the obvious physics does not help
  here. An energy-gain law integrated along the density profile misses most of the variation
  between configurations, because beam loading (the bunch's own charge weakening the
  accelerating field) sets the gain more than the density does, and past the point where the
  bunch outruns the wave (dephasing) the gain stops. The Hard objectives give our numbers;
  finding the physics that *does* help is open. See Esarey *et al.* (below) for both effects.
- **The data the approach needs.** Every training sample is a cluster job. A model that reaches
  a useful score on 300 simulations is a different proposition from one that needs 3000, and the
  difference is weeks of compute for whoever builds the next campaign.
- **The behaviour outside the sampled region.** A surrogate is most useful exactly where nobody
  has simulated yet. The dataset therefore withholds a band of plasma densities and a set of
  positions along the accelerator, so that extrapolation can be scored rather than hoped for.

There is a fourth problem in the data that is genuinely open rather than merely hard. Going
**backwards** — from a measured bunch to the settings that produced it — is not a well-posed
map: different settings give similar beams. Our own reference ceiling on that task is R² 0.9909
and not 1.0, and that gap is degeneracy in the physics, not model error. A model that returns
one confident number is not answering the question, however good its R² looks; the honest output
is a distribution, and whether that distribution is calibrated is the real score.

## Current state / prior work

The dataset and the reference models come from the PALLAS design study and the surrogate-model
work that followed it.

- **Forward surrogate (settings → bunch).** Our internal PCA plus gradient-boosting emulator
  (not published) reaches R² 0.9744 on median energy and 0.9420 on charge on the frozen
  test split. Scored along the whole trajectory on this pack's own grid and scorer, the same
  model reaches a mean R² of 0.9069 inside the plasma and 0.8840 in the drift after it.
- **Inverse model (bunch → settings).** Our five-member mixture-density ensemble (in review)
  reaches a mean
  R² of 0.9100 against an oracle ceiling of 0.9909. On about 63 % of shots the reference
  posterior puts more than a fifth of its weight on a second, well-separated solution — which is
  the degeneracy, made visible.
- **The part that is nearly solved, and the part that is not.** Predicting the bunch at the *end*
  of the accelerator from Campaign A's four settings is essentially done: a thirty-line network
  reaches R² ≈ 0.99, which is why that one is a warm-up rather than a scored task. The same
  question on Campaign B's density-profile settings is not solved — our internal forward model
  reaches 0.9744 on median energy and 0.9420 on charge, but only 0.8036 on the energy spread —
  so the forward direction is scored there, on the `test_direct` set.

All reference numbers quoted here are shipped inside the data pack, in the `scores` section of
`pack_reference.json`; the notebook quotes the same numbers, so a team can check each one against
the file.

## References

- G. Kane *et al.*, "Surrogate models study for laser-plasma accelerator electron source design
  through numerical optimisation", *Machine Learning: Science and Technology* **7**, 030502
  (2026). doi:10.1088/2632-2153/ae6603
- P. Drobniak *et al.*, "Random scan optimization of a laser-plasma electron injector based on
  fast particle-in-cell simulations", *Phys. Rev. Accel. Beams* **26**, 091302 (2023).
- J. Derouillat *et al.*, "SMILEI: a collaborative, open-source, multi-purpose particle-in-cell
  code for plasma simulation", *Comput. Phys. Commun.* **222**, 351 (2018).
  doi:10.1016/j.cpc.2017.09.024
- E. Esarey, C. B. Schroeder and W. P. Leemans, "Physics of laser-driven plasma-based electron
  accelerators", *Rev. Mod. Phys.* **81**, 1229 (2009).
- A. Döpp *et al.*, "Data-driven science and machine learning methods in laser-plasma physics",
  *High Power Laser Sci. Eng.* **11**, e55 (2023). doi:10.1017/hpl.2023.47

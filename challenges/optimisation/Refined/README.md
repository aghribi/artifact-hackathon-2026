# Refined — an arbitrary lattice, tuned

A second pass at the tuning pipeline, built to take **any** Cheetah lattice rather
than CLARA specifically. `MBRL_opt/`, `MFopt/` and `optics_matching/` between them
work very well on one machine and are wired to it in a few dozen places: magnet names
in configuration files, a VBC chicane special-cased in four modules, a matched optics
family baked against one lattice file. This is the same method with those assumptions
pulled out into one place the user fills in.

## The shape of it

```
stage 1 — precompute          stage 2 — tune
  (minutes to hours,            (seconds to minutes,
   once per lattice)             once per request)

  lattice.json  ─┐
  machine.yaml  ─┼─► optics family ─────► SVD of the response ─► optimise
  beam.npz      ─┘   k1(sigma_x,sigma_y)   at the chosen optics    the machine
                     + provenance
                     ───────────────
                     artifact.npz
```

Two scripts, because the two halves have completely different costs. The optics
family is a grid of matched tunes and takes hours; it depends only on the lattice, so
it is computed once and reused for every request, every episode and every study.
Everything downstream of it is per-request and cheap enough to do live.

| | |
|---|---|
| `describe.py` | the interview: read a lattice, ask what it cannot say, write the spec |
| `sensitivity.py` | which knobs invalidate the family, ranked, so the axes are chosen by measurement |
| `lattice.py` | `Line`: any spec, tracked in torch, batched and differentiable |
| `match.py` | the matcher: a whole grid of requested spots solved in lockstep |
| `precompute.py` | stage 1: spec → `artifact.npz` |
| `artifact.py` | `Family`: read the artifact back, look a request up |
| `machines.py` | what a method drives: `SimEnvMachine`, for whatever machine the spec describes |
| `methods.py` | the tuning methods: `optics`, `steer`, `mpc`, `bo` |
| `calibrate.py` | measure what a machine needs that only a family can reveal |
| `optimise.py` | stage 2: artifact + a request → a tuned machine |

and three packages it carries with it:

| | |
|---|---|
| `machine_spec/` | the `MachineSpec` itself, and the lattice/beam/aperture readers |
| `rl_framework/` | `SimEnv` and `RealEnv` behind one interface, and the CLARA control-system driver |
| `screen.py` + `image_processing.py` | a camera frame → four numbers, the measure the real screens are read with |

**This directory is self-contained.** Everything the pipeline imports is in
here, so it can be copied out as a unit and the only path setup any module
needs is `Refined/` itself. There is exactly one deliberate exception:
`data/clara/lattice_data/utils.py`, which holds CLARA's magnet calibrations and
reference energies, is left where it is and found by walking up. It is shared
with `score.py` — the benchmark this pipeline is graded against — and two
copies of a calibration that could drift apart is the precise failure
`hardware_CLARA.py` exists to prevent. The lattice, beam and aperture files the
specs point at are likewise data, and `MachineSpec.load` already finds them by
walking up from the spec.

```bash
python -I describe.py lattice.json --out machines/mine.json   # once per machine
python -I sensitivity.py machines/mine.json                   # what to scan over
python -I precompute.py machines/mine.json --pilot            # a coarse dry run
python -I precompute.py machines/mine.json                    # the real thing
python -I calibrate.py machines/mine.json artifacts/mine.npz --apply
python -I optimise.py machines/mine.json artifacts/mine.npz --method steer
```

## Why the machine description exists

A lattice file says what the elements are. It does not say what they *mean*, and the
difference is exactly what has to be hand-written:

- **Which magnets are knobs.** A dipole in a dogleg is geometry; the four dipoles of
  a bunch compressor are one knob. Nothing in the file distinguishes them.
- **Which move together.** CLARA's VBC is four dipoles on one power supply with
  alternating signs, and setting the angle has to set the pole faces with it or the
  edge focusing is wrong. That is a fact about the hardware.
- **Which are non-linear.** Sextupoles, the chicane, RF phases: the knobs whose
  effect is not a matrix, so they cannot go in a linear response and have to be
  searched. The user names them.
- **What the limits are.** Power supply ranges are a property of the machine.

`describe.py` infers everything it honestly can — element types, screens, BPMs,
apertures, obvious knobs — and leaves the rest as commented stubs with the reasoning
spelled out. It is a starting point to edit, never an answer.

## What is carried over

Everything here was learned the hard way in `MBRL_opt/` and is kept deliberately:

- **One source of truth for the lattice.** The model the optimiser plans against is
  built from the same file and the same options as the machine it is tuning, never
  from a second copy. Three separate bugs came from a stale duplicate.
- **Clear the transfer-map cache on every write.** Cheetah keys its cached maps on
  beam energy, not on the strengths they were built from, so a batched evaluation
  leaves a batched map behind and the next unbatched track silently returns the wrong
  shape.
- **Steps measured in effect, not in units.** A step is whatever moves the readout by
  a set fraction of the requested spot, calibrated per knob from the measured
  response. No single physical step size suits a corrector at the injector and one a
  metre before the screen.
- **Size and position are separated.** Quadrupoles match the spot; correctors nearest
  the screen place it, in one linear solve. Together in one optimiser, a quadrupole
  will fix a position error by distorting the optics.
- **Objectives are not clipped at the benchmark's scale.** An untuned machine sits
  far past where the score stops counting, and a clipped cost is flat exactly where
  the optimisation happens.
- **Apertures get a straight-through gradient.** Value from the hard edge, gradient
  from a sigmoid. Blurring the edge outright is badly biased.

## What building it taught us

Four things that were not obvious and each of which was, at first, silently wrong.

**A family has to be smooth, not just correct.** The fast way to match a grid is to
sweep the knob box once and start every node from whichever sample came closest.
It matches more nodes than anything else and produces a family that cannot be
interpolated: neighbouring requests land in unrelated basins, and on CLARA
adjacent nodes came out up to 20 m⁻² apart — the entire span of the knob — with
every node individually correct. Asked for a spot between two of them the family
returned the average of two unrelated answers, which delivered 3.9 mm where it
promised 1.9. `precompute.solve_coarse` now seeds **one** node and grows outward
by continuation, so the whole grid stays on one branch. `_roughness` is printed at
every level, because a grid that matches every node and is rough is worse than one
that misses a few and is smooth: the first lies and the second says so.

**Transmission is a property of the solution, not a constraint on it.** Carrying
`2(1 - T)` as a third residual, as a least-squares matcher reasonably does, makes
a Gauss-Newton step aim at full transmission — which is usually unreachable, so
the solve oscillates forever around a target that does not exist. The sizes were
matched to 5e-7 while the reported residual sat at 1-2.

**The lattice's own tune is not a working tune.** CLARA's ships one that scrapes
26 % of the beam at a single aperture before anything is tuned, and the benchmark
weights transmission at twice the size error — so that one aperture is worth about
0.5 of score, more than any amount of clever tuning downstream. `repair_optics`
fixes the standing optics first, 0.741 to 0.998. It uses CEM rather than gradient
descent, and that is not a preference: the scraping is Cheetah's own aperture
elements, whose survival mask is a hard step, so the gradient of transmission with
respect to an upstream quadrupole is exactly zero. 120 Adam steps moved it by 0.005.

**A sweep that samples cleverly can miss the obvious.** `sensitivity.py` used a
geometric ladder around the present setting, which resolves a sharp knob beautifully
and skips the middle of its range. CLARA's compressor holds the beam above
|0.115| rad either side of zero and loses four fifths of it in between, so the scan
declared the whole range usable and the first precompute spent four grid points of
five on tunes nobody can run — 16 % feasible. With plain linear coverage unioned in,
the range came out as the real band and the same precompute gave 82 %.

## A future improvement: differentiate the match instead of sampling it

The grid is exponential in the number of scan axes, and the reason is that the
matched optics is treated as a black box that can only be sampled. It is not. The
match solves a residual to zero, so the implicit function theorem gives its
derivative with respect to anything held fixed, in closed form, from quantities the
matcher already computes and currently throws away:

```
r(k, p) = 0        =>        dk*/dp = -J_k⁺ · ∂r/∂p
```

`k` is the matching quadrupoles, `p` a non-linear knob, `J_k` the Jacobian the
least-squares solver builds at every iteration. `∂r/∂p` is one extra column per
knob, from the same batched track. The derivative therefore costs a few percent on
top of a match that was happening anyway.

Three things fall out of it:

- **Hermite interpolation instead of linear.** Error per axis goes `O(h²)` to
  `O(h⁴)`, so the same accuracy needs roughly the square root of the points. That
  is what makes a third or fourth axis survivable.
- **The real axis count.** Stack `dk*/dp` over every candidate knob and take the
  SVD. If the rank is two, then every one of those knobs moves the matched optics
  along the same two directions, and the grid spans *those*, not the knobs. This is
  the quantitative version of what `sensitivity.py` estimates by sweeping, which
  reports per-knob magnitudes and cannot see that two knobs are redundant.
- **Reparameterising onto the incoming beam.** On CLARA every candidate non-linear
  knob sits strictly upstream of the four matched quadrupoles, so it can only
  influence the match through the beam arriving at the final focus. The complete
  mediator is that beam's second-moment matrix and energy spread — a handful of
  numbers, fewer than the knobs. A family indexed by the incoming beam also
  survives configuration changes nobody scanned.

The caveats are real but bounded: a quadrupole pinned at a limit has a one-sided
derivative and the solution is only piecewise smooth, `J_k` goes ill-conditioned
near the edge of feasibility (which is physical, not numerical), and with more
quadrupoles than constraints the matched point is not unique, so the min-norm gauge
has to be stated rather than left to wherever the solver happens to stop. The
cheapest first step is the SVD of the stacked sensitivity: it decides whether a
dense grid is tractable at all, and it can be done on the nodes that already exist.

## A family has to be smooth *and* cover what gets asked for

The two pull against each other and getting either alone is worse than useless.

Continuation keeps the grid on one branch, which is what makes it interpolable —
and it can only extend the branch it starts on. On CLARA that branch stops at about
0.5 mm vertically, so a third of the grid came back unmatched, and it was precisely
the third the machine is asked for: `SimEnv` requests vertical spots of 0.11 to
0.39 mm. Every lookup fell back to the nearest feasible node and the family quietly
answered a different question than the one put to it.

A Sobol sweep of the knob box reaches that corner easily — vertical spots down to
0.023 mm — because it has no attachment to any branch. It also has no idea the rest
of the grid exists, which is how the first version produced neighbouring nodes
20 m⁻² apart.

So `solve_coarse` uses each for what it is good at, in three stages:

1. **continuation** outward from one seeded node, for coherence;
2. **a Sobol rescue** of whatever that could not reach, for coverage;
3. **a coherence relaxation** that re-matches every node from the mean of its
   neighbours and keeps the result only when it converges *and* sits closer to
   that mean — so rescued nodes get pulled onto the local branch and nodes already
   agreeing with their neighbours do not move.

Only the coarsest level does this; the finer levels inherit the branch choices
through interpolation and refine locally. That is why the ladder starts at 13 × 13
and not 7 × 7: at 7 × 7 only four nodes land in the sub-millimetre corner, which is
not enough to discover that it is reachable at all.

## Testing it on a machine that is not CLARA

For a while the lattice-agnosticism test was `LineMachine`: a `Line` perturbed
into a machine, so the pipeline could be exercised without CLARA. It was a poor
test and it is gone. It shared its code with the planning twin, so a model-based
method faced no model error at all — the thing `mpc` exists to absorb — and it had
no shot noise: identical settings gave identical readings, where `SimEnv`'s differ
by about 5 %.

`SimEnv` now takes a `MachineSpec` too, so the test is the honest one: point the
whole pipeline at a different accelerator. `machines/clear.json` is CERN's CLEAR
line, and `describe.py` → `sensitivity.py` → `precompute.py` → `optimise.py` runs
on it end to end — 289 of 289 nodes matched in 4 seconds, and `steer` takes the
score from 1.42 to 0.28 in four shots.

That spec is **synthesised** and says so in its own notes: CLEAR ships as a
lattice rather than a machine, with every quadrupole at zero, no beam
distribution, no calibration and a 0.512 m half-size screen that is a converter
default. The beam, the limits and the chamber were invented to give the pipeline
a second machine to run on. It tests that the code is general; it is not a claim
about CLEAR.

## Status

Working end to end on two machines. `describe.py` → `sensitivity.py` →
`precompute.py` → `optimise.py`, with `optics`, `steer`, `mpc` and `bo` against
`SimEnv`.

## Open: the VBC stage, and the angle the family scans

**This blocks tuning the real CLARA, and it may cost a `precompute.py` re-run.**

The chicane's bend comes from the dipole currents; its *geometry* comes from a
translation stage on `CLA-VBC-MAG-TRANS-01:TGTPOS`, with `:MOVING` to say when
it has stopped. The two are one setting in two places and only the first was
wired. `cla_vbc_chicane` is the optics family's only scan axis and `optics`
sets it on shot one, so this is on the critical path rather than off to one
side.

The numbers do not currently reconcile. TGTPOS is in mm, reads 372.98 at zero
offset and has a floor at 330, so there is 42.98 mm of safe travel. The lattice
says the middle of the chicane has to sit `L sin(theta)` off axis with
`L = 1506.7 mm` from `vbc_drift_1`:

| chicane angle | offset needed | TGTPOS | against the 330 floor |
|---|---|---|---|
| 0.1215 rad (family minimum) | 182.6 mm | 190.4 | −140 |
| 0.1317 rad (lattice maximum) | 197.9 mm | 175.1 | −155 |

Every tune in the shipped family would drive the stage about 150 mm past its
limit. Equivalently, 42.98 mm of travel is a bend of 0.0285 rad against the
0.1215–0.1317 the optics use — short by a factor of 4.3.

One of three things is wrong, and the useful next step is to find out which:

1. **the stage does not carry the full `L sin(theta)` beam offset** — most
   likely; it may move the dipoles longitudinally, or carry only part of the
   geometry;
2. the lattice's angle is not the quantity the stage tracks;
3. the zero or the floor is not what we think.

Until then `VBC_STAGE_POSITION` in `hardware_CLARA.py` raises and explains
itself, and a write to any VBC dipole is refused rather than half-applied. The
stage is otherwise wired: readable, movable, waitable, with the floor enforced
in `SET_REAL_VBC_STAGE`. The PVs are not on the virtual accelerator, so none of
it has been exercised against hardware.

Two consequences worth holding on to:

- **If the reachable bend really is ~0.0285 rad, the family is unusable on the
  real machine as built.** It was scanned over 0.1215–0.1317 and would need
  rebuilding over a range the stage can reach — a `precompute.py` re-run, which
  took 2792 s for the shipped artifact. Settle the mapping first.
- **The model does not represent the stage either.** `vbc_drift_1` is fixed at
  1.506747 m while the angle varies, so the path length should shrink about
  2 mm across the family's band and does not. Small, but the artifact rests on
  a slightly wrong geometry whatever the hardware turns out to do.

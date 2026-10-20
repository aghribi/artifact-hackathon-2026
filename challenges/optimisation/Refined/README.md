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
| `describe.py` | read a lattice, write a draft `machine.yaml` for the user to edit |
| `precompute.py` | stage 1: lattice + spec → `artifact.npz` |
| `optimise.py` | stage 2: artifact + a request → a tuned machine |

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

## Status

Being built. `describe.py` first, because nothing else can be written until the
machine description has a shape.

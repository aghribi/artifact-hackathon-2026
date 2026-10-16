# MBRL_opt — the model-based RL stage of the tuning pipeline

Getting a `SimEnv` episode — a new day on CLARA, with its own beam, its own survey
error and a request for a spot at `cla_feh_dia_scr_01` — onto that spot in about six
shots, without ever moving a screen.

This is the last stage of the team pipeline and it sits on the other stages rather
than replacing them: it takes the reference optics and the SVD control rankings from
[`MFopt`](../MFopt) / [`optics_matching`](../optics_matching), and reads its beam with
the screen measure from [`image_processing`](../image_processing). What is added here
is the twin and the tweak.

```
request (mu_x, mu_y, sigma_x, sigma_y)
        │
        ▼
1. reference optics      MFopt.optics_for_target  ──  no shots, ~40 s of model time
     46 quadrupole k1s that give the requested size
        │                                                         1 shot to load
        ▼
  (steer               SVD: correctors vs BPMs — implemented, off by default,
                       because on this machine it makes things worse. See below.)
        │
        ▼
2. control set           SVD: leading quadrupoles, plus the
     sextupoles, RF phases and chicane                             no shots
        │
        ▼
3. match the size        differentiable twin + CEM/Adam            0–12 shots
     fit what the machine is, plan against it, re-read the camera
     scores the spot size and transmission only
        │
        ▼
4. place the spot        SVD: final correctors vs readout centroid  1 shot
     two-row least squares, not a search
```

**Size and position are separated on purpose.** The focus works only on the
quadrupoles and scores only the size; where the spot lands is settled afterwards by
the two correctors nearest the screen, in a single linear solve. Put the two in one
optimiser and a quadrupole will happily fix a 2 mm position error by distorting the
optics — which it can do, and which costs the size the focus just bought. Separating
them took the score from 2.66 in 25 shots to 0.29 in 6 on the seed they were first
compared on.

**The BPM steering is off by default, on measurement.** Over 12 episodes, running it
scored 1.485 and skipping it 0.525, for 3.5 fewer shots. The stage is not broken;
there is nothing here for it to do. The reference optics already arrives with 99 %
transmission and all 31 BPMs reading, so there is no beam to recover — and flattening
the *measured* orbit steers the beam onto the BPM centres rather than the magnetic
axis, because each BPM carries its own 30 µm centring error. Ten correctors chasing
62 such readings put a real distortion on a trajectory that was fine. It is kept, and
`steering_shots` turns it back on, because a machine that really has lost the beam —
`score.py`'s bad set points rather than SimEnv's — is exactly what it is for.

## Why it is shaped like this

**The lookup does most of the work, for free.** `optics_for_target` re-matches the
final focus on an error-free copy of the model and hits the requested *size* to a
fraction of a percent at 99 % transmission, without touching the machine. Stage 4
therefore is not solving the tune — it is correcting the difference between the
machine the optics were matched for and the one the episode dealt.

**The lookup matches size, not position.** The request also names a spot anywhere
within 2 mm of the screen centre, which is about eight requested spot sizes and most
of the score. That is what the final correctors in the focus set are for, and it is
why the centroid SVD alone is not enough: it ranks `cla_feh_mag_hvcor_01/02` near the
bottom precisely because they move nothing but the last screen, which is exactly the
property wanted here.

**Steps are measured in effect, not in units.** Before the focus starts, one batched
track gives the response of all four measured moments to every control knob, and a
step is redefined as the move that changes the reading by half a requested spot
(`trust_effect`). No fixed physical step works: 0.1 mrad is a large kick from a
corrector near the injector and nothing from the one a metre before the screen — and
it is the second that has to carry the spot 2 mm. Before this was fixed the planner
plateaued at a mismatch of 8 because it could not reach the requested position.

**The planner is anchored on the measurement.** It never asks the twin where the beam
is, only what *changes* if a knob moves:

```
predicted(delta) = measured + [twin(knobs + delta) − twin(knobs)]
```

A model wrong by a constant still predicts the effect of moving a quadrupole, and
anchoring cancels the constant exactly.

**The beam is read off the camera.** `screen.measure` wraps the measure written for
the real CLARA captures, so the agent is driven by the same quantity in simulation as
it would be on the machine. The benchmark still scores the particle moments; the agent
never sees them.

## Running it

```bash
cd challenges/optimisation
uv pip install --python .venv/bin/python \
  -r RL_framework/requirements.txt -r nominal_model/requirements.txt

.venv/bin/python -I MBRL_opt/smoke_test.py                      # ~2.5 min, 23 checks
.venv/bin/python -I MBRL_opt/evaluate.py --episodes 8 --workers 8
.venv/bin/python -I MBRL_opt/evaluate.py --episodes 8 --fit-misalignments
```

Episodes are independent and each pins itself to one thread, so `--workers` is close
to linear. An episode is two to three minutes; most of it is the twin.

## The files

| File | What it is |
|---|---|
| `agent.py` | the four stages and the shot budget |
| `twin.py` | `CheetahTwin`: the differentiable model, and `readings()` reproducing `SimEnv._track`/`_expose` exactly |
| `system_id.py` | fits the beam (and optionally the survey error) to the episode's shots, through Cheetah |
| `planner.py` | truncated-SVD steering, CEM + Adam focus, anchored on the last reading |
| `knobs.py` | the control vector, its physical scales and its effect-calibrated steps |
| `screen.py` | the robust camera measure, wrapping `image_processing` |
| `config.py` | `MBRLConfig` |
| `baselines.py` | the pipeline with its later stages removed |
| `dyna.py` | a `CheetahTwin` dressed as the environment, so episodes cost no beam |
| `features.py` | the episode-independent view a policy sees: SVD rank slots, not named magnets |
| `policy.py`, `train.py` | the Dyna layer: clone the planner inside the twin |
| `evaluate.py`, `smoke_test.py` | the comparison and the checks |

## Results

**100 episodes**, seeds 1234–1333, at SimEnv's defaults (`beam_variation` 1.0,
`misalignment_variation` 1.0, shot jitter, apertures, VBC on, L4H off). Every method
sees the same episodes. Score is the mean `info["benchmark_score"]`, which is
`score.Report.calc_score` for one episode — mismatch in requested spot sizes, plus
`TRANSMISSION_WEIGHT` for the bunch that never arrived, plus the shot cost. Lower is
better. The hit columns are `score.HIT_TOLERANCES`.

| method | score | median | shots | screen switches | wall s/ep | <50% | <25% | <10% |
|---|---|---|---|---|---|---|---|---|
| **full pipeline** | **0.77** | **0.57** | 6.5 | **0** | 1086 | 86 % | 60 % | 21 % |
| optics + placement | 1.00 | 0.81 | 2.0 | **0** | 7.5 | 62 % | 36 % | 8 % |
| reference optics only | 6.79 | 6.93 | 1.0 | **0** | 5.0 | 3 % | 2 % | 0 % |
| do nothing | 9.65 | 10.00 | 0 | **0** | 0 | 0 % | 0 % | 0 % |

Against the four figures of merit:

- **Screen switches: zero, always.** Not a tuning result but a design one. All 31
  BPMs sit upstream of `cla_feh_dia_scr_01`, and BPMs do not intercept the beam, so
  the readout screen can stay in for the whole episode. At 30 s a swap, the pipeline
  never spends any of it. `EpisodeResult.screen_switches` counts them so this stays
  true if the design changes.
- **Shots: 6.5 on average**, and 2 of those are fixed overhead — one to load the
  optics, one to place the spot. The focus takes the other 4.5 and often takes none
  at all, because the reference optics frequently lands the size inside the
  shot-to-shot jitter already. At `EVAL_BUDGET` 50 this costs 0.06 of the score.
- **Match: 0.77 mean, 0.57 median**, 86 % of episodes inside half a requested spot
  and 21 % inside a tenth. Beam delivered on 100 % of episodes.
- **Wall clock: ~18 minutes per episode** of model time on a 64-core container at
  about half a laptop's per-core speed (the 1086 s above is per-episode wall under
  50-way contention). None of it is beam time. The optics lookup is ~40 s and the
  rest is the twin: the focus is the expensive stage, which is why it is worth
  stopping it as soon as it is inside the noise.

The jump from 6.79 to 1.00 is one shot of placement, and the jump from 1.00 to 0.77
is the model-based focus. Both matter, and they are very different kinds of thing:
the first is a two-row linear solve, the second is 4.5 shots of search against a
fitted twin.

### What each stage is worth, and the steering ablation

12 episodes, seeds 1234–1245, same conditions:

| variant | score | median | shots |
|---|---|---|---|
| optics + focus + placement (default) | **0.53** | 0.40 | 6.0 |
| optics + steering + placement | 0.63 | 0.49 | 5.9 |
| optics + steering + focus + placement | 1.49 | 0.73 | 9.5 |
| optics only | 6.99 | 7.91 | 1.0 |

Every variant that includes the BPM steering is worse than the one without it, and
pays 3.5 extra shots for the privilege. That is why `steering_shots` defaults to 0;
see the note above for why the stage has nothing to do on this machine.

The misalignment fit (`--fit-misalignments`) was worth about a third of the score on
the earlier, coupled version of the pipeline (1.44 against 0.95 over 32 episodes). It
has not been re-measured since the split and the numbers above are with it off.

Reproduce with `evaluate.py --episodes 100 --seed 1234 --workers 50`. Numbers are in
`results_100.json` and `results_ablation.json`. Give `--workers` at least as many
slots as there are episodes, or the leftovers queue for a whole extra wave.

### Where the wall clock goes

One episode, measured uncontended, with the stages timed separately:

| | laptop (12 core) | apml1 (64 core) |
|---|---|---|
| per episode, mean | **~270 s** | **557 s** |
| range over 4 seeds | 247–287 s (4–6 shots) | 391–739 s (4–7 shots) |

apml1 is *slower* per episode — about half the per-core speed. It wins only by
running fifty at once, which is why a 100-episode study takes ~40 minutes there and
would take seven hours here. Pick the box by how many episodes you want, not by how
fast you want one.

The spread is almost entirely shot count: roughly 150 s of fixed cost (the optics
lookup) plus ~70 s per focus shot on apml1. A 4-shot episode is 391 s and a 7-shot
one 739 s.

Within an episode, in rough order (seed 1234 on apml1, 470 s):

| stage | s | what it is |
|---|---|---|
| CEM | 208 | 128 candidates × 4 rounds per exploring shot |
| optics lookup | 105 | 278 full-lattice tracks, see below |
| system ID | 80 | Adam through Cheetah onto the shot history |
| Adam (plan polish) | 66 | 12 gradient steps per shot |
| Jacobians | 3 | one batched track each for calibration and placement |

**The optics lookup is 278 serial tracks** of the whole 655-element line: 93 for the
quadrupole beam-size SVD (46 quads, central differences), 113 for the corrector
centroid SVD (56 planes), ~71 for the `least_squares` re-match of the four
final-focus quadrupoles, and one at the end. It is cached per target and every
episode draws a new one, so every episode pays in full.

Two easy savings sit in `MFopt/optics_env.py` rather than here. **41 % of the lookup
builds a corrector ranking this pipeline no longer uses** — steering is off and
`focus_correctors` is 0, so `optics.correctors` is only read for knob names in a loop
that never runs. And the quadrupole ranking barely changes between nearby requests,
so caching it across targets and re-running only the four-quad match would take the
lookup to about a quarter of its cost. Together that is most of the 105 s.

After those, CEM is the thing to attack, and `explore` — running it only when the
tune is actually stuck — is the lever already in place.

### Cost per episode

Measured on seed 1234: 25 shots on the machine, 290,000 particle-tracks. The optics
lookup spends 2.8 M on its own error-free twin, and the agent's twin spends 30.7 M
across 611 calls — most of them batched, since a CEM iteration is 128 candidates in
one call. The machine is about 1 % of the arithmetic, which is the whole point: a
shot is the only thing that costs beam time.

## Things worth knowing

**The twin has to be the environment's own lattice.** It is built through
`xsuite_model.cheetah_segment`, from `data/clara/lattice_data/CLARA_cheetah.json`,
with apertures on — not from the copy at `RL_framework/FEBE_full.json`. The two were
identical until the VBC was turned on and only one was updated; a twin on the stale
copy disagreed with the machine by 1.5 mm at the readout and the planner spent the
episode optimising a different accelerator. `smoke_test.py` holds the two to 0.0001 µm
on an error-free machine, and that check is the one to run first after any change
upstream.

**Two aperture subtleties, both load-bearing.** Whether a particle is inside the beam
pipe is a step function, and half the beam is scraped in the FEA arc, so the objective
is a staircase with no slope. Blurring the wall is badly biased — at the true latent
it moved the predicted loss from 0.00 to 114.7 and the fit walked away chasing it — so
`twin._aperture` takes the *value* from the hard edge and the *gradient* from a
sigmoid. And the 282 aperture elements break Cheetah's map fusion, which is why a
track costs 90 ms rather than 37.

**Batching is not free for these knobs.** A batch of candidate RF phases is a batch of
different beam energies, so nothing downstream can be fused: at 128 candidates the six
quadrupoles cost 287 ms and all seventeen knobs 1675 ms. A candidate is ~13 ms against
~160 ms for a gradient step, so CEM still wins, but only enough to be worth running
when the tune is stuck — hence `explore`.

**Do not clip the planner's cost at the benchmark's scale.** The benchmark stops
counting at ten requested spot sizes; an untuned machine sits at seventy. A cost
clipped at ten is flat across the whole region where tuning happens.

## Limits, and what to try next

**The size residual is the binding constraint.** Position comes out to a few tens of
microns, but when the episode's beam is vertically mismatched the readout `sigma_y`
can end several times the request and the focus set cannot pull it back inside the
budget. The most promising fix is to close the loop on stage 1: re-run
`optics_for_target` against the beam the system identification has *fitted*, rather
than the nominal beam it assumes, and reload. That is a second 40-second lookup and no
extra shots.

**The lookup sometimes misses.** On one of the two smoke-test seeds it returned optics
21.8 % off the requested size in its own model, which caps what the tweak can do.
That is `MFopt`'s to fix; `EpisodeResult.optics_error` reports it per episode.

**Steering is nearly free and nearly useless here.** The reference optics already
delivers beam with every BPM reading, so there is no beam to recover, and the orbit
bottoms out around 550 µm rms — a handful of correctors against a survey error on 46
quadrupoles. It would earn its place on a machine that had lost the beam.

**Turn the misalignment fit on.** It is off by default so the aligned twin is the
baseline, but a survey error does things no launch offset can imitate.
`--fit-misalignments`.

**The Dyna layer is built but untrained.** `dyna.py` dresses a `CheetahTwin` as the
environment so the agent runs against it unmodified; `train.py` rolls episodes
through it, records what the planner decided, and `policy.py` clones those decisions
into a network that proposes the focus move as a warm start. All of it is verified
end to end — a twin episode converges 9.75 → 0.67 and yields usable pairs, and the
cloning fits — but no policy has been trained at scale, and the case for one is now
weaker than it was: the focus is down to ~4.5 shots and often zero, so a policy would
buy wall-clock rather than shots or score. Worth revisiting only after the size
residual below.

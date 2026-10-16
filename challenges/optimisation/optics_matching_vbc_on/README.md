# Eiad's optics nodes with apertures and the VBC on

This folder checks the transmission of the 144 matched-optics nodes in `optics_matching/`
(Eiad Hamwi: 12x12 target rms sizes, 0.1 to 20 mm at `cla_feh_dia_scr_01`, all 46 quadrupole k1).
It then rematches the nodes for the VBC on.

## Setup

- Cheetah `SimEnv` with apertures on, the nominal measured beam, and no misalignments, jitter or ripple. No magnet is clipped to ±100 A.
- Every Drift, Quadrupole and Dipole is tracked at second order, with second-order dipole edges.
- `CLARA_cheetah_second_order.json` is the challenge lattice converted to second order. `second_order_lattice.py` rebuilds it.
- Transmission is the fraction of the launched bunch alive at the FEH screen.
- Sizes are the rms, about the centroid, of the surviving particles at the screen.

## 1. Transmission of the original nodes

**Scripts:** `evaluate_nodes.py` (4 parallel workers, 50k particles, about 3.5 min for 432 runs) and `plot_matrix.py`.

**Results** (`results/transmission_{vbc_on,vbc_off,design}.npz`, `results/summary.json`):

- **VBC on, 118.5 mrad (master):** median 92.7%. The nodes lose beam quickly once σy goes above about 2 mm or σx reaches 20 mm, down to 17% at the corner. The nodes were matched with the VBC off.
- **VBC off:** about 97% at almost every node, with a minimum of 60% at 20×20 mm.
- **Eiad's own anchor lattice** (VBC off, L4H off): about 97%, with a minimum of 63%.

In Cheetah the original nodes do not reproduce their target sizes, even with the VBC off. They were validated in Xsuite on the whole bunch, without apertures.

## 2. VBC setting

The model's R56 values compare with the two papers as follows:

| | Model | Paper |
|---|---|---|
| FEBE arc | +7.75 mm | +7.7 mm (nominal) |
| VBC at 118.5 mrad | −46.5 mm | — |
| Maximum compression into FEBE | — | −57.6 mm (100 pC) |

The papers are Jones et al., IPAC'26 THP2100 and Williams et al., arXiv:1205.3887.

The rematch follows the paper: the VBC is set to **131.7 mrad**, which gives R56 = −57.6 mm.

Not changed here:

- The model energy at the VBC is 181.6 MeV, while `utils.py beam_energy` gives 196.6 MeV.
- L02 and L03 run at 23° and 8°. The paper quotes +12° to +30°, with both cavities at the same phase.

## 3. Rematch with the VBC on

**Script:** `rematch.py`, followed by `plot_rematch.py` and `plot_quads.py`.

**Method:**

- Eiad's 42-quad prefix is kept.
- The 30 quads after the VBC (S06, S07, FEA, FEH) are refitted per node with L-BFGS, through differentiable Cheetah tracking with 3k particles.
- Loss = ln(σx/target)² + ln(σy/target)² + 5·(1 − T).
  - The sizes use the particles that survive the hard apertures.
  - T is a soft transmission: each aperture becomes a sigmoid.
- Every node is then checked in `SimEnv` with 50k particles.
- 144 nodes take about 40 min on 4 workers. Set `VBC_ANGLE` to rematch at a different VBC angle.

**Output:** `results/rematched_grid_vbc_on.npz`. It has the same layout as `optics_matching/results/validated_grid.npz`:

- `k1`: (12, 12, 46), indexed [σx index, σy index, quad]
- `quadrupoles`, `spot_sizes_m`
- `transmission`, `actual_sizes_m`, and the `_eiad` equivalents for the original nodes at the same VBC

Per-node detail is in `results/rematch_results.json`.

**Results** at 131.7 mrad, from 144 nodes:

| | Eiad's nodes | Rematch |
|---|---|---|
| Median transmission | 89% | 99% |
| Median size error (worst plane) | 285% | 1.6% |

- 96 rematched nodes have T ≥ 95% and a size error ≤ 5%.
- The large-size corner (σ of 12 to 20 mm) still fails, with size errors of 10–54%. The 12×20 mm and 20×20 mm nodes are at 3–5% transmission.
- The size knobs are still FEA Q15/Q16 and FEH Q1/Q2.
- The VBC correction is mostly S06 Q01/Q02 (up to 2.6 m⁻²) and FEH Q1.
- The other quads jitter by a few tenths of a m⁻² from node to node. They are nearly degenerate.
- 64 nodes need FEA Q16 or FEH Q1/Q2 beyond ±100 A. Eiad's family needs this at 62 nodes.

## Rerun (from `challenges/optimisation`)

```
python optics_matching_vbc_on/second_order_lattice.py
python optics_matching_vbc_on/evaluate_nodes.py optics_matching_vbc_on/results 50000 4
python optics_matching_vbc_on/plot_matrix.py optics_matching_vbc_on/results talk_figures/optics_nodes
python optics_matching_vbc_on/rematch.py optics_matching_vbc_on/results 4 3000
python optics_matching_vbc_on/plot_rematch.py optics_matching_vbc_on/results optics_matching/results/validated_grid.npz 0.13169935 talk_figures/optics_nodes
python optics_matching_vbc_on/plot_quads.py optics_matching_vbc_on/results optics_matching/results/validated_grid.npz talk_figures/optics_nodes
```

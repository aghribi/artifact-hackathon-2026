# TODO
Antonio - please write some things about CLEAR as well

# Scientific Case — Optimisation (CLEAR / CLARA)
CLARA is a linear electron accelerator with a large number of diagnostics. The machine is frequently power cycled and modified for user experiments. Hysteresis and machine drift
affect operating points and much time is spent tuning and optimising for requested user setups, which can involve a wide range of beam parameters.
In the ideal case, operators would be able to specify a set of target beam parameters at the interaction point and have the machine automatically tuned to meet those parameters.

The sim-to-real gap complicates this considerably. Dark current and wakefield induced features are present in the beam, space charge effects dominate in the injector, and coherent
synchotron radiation effects are present in the bunch compressor. With rapid simulation software like Cheetah, some of these effects are not modelled and the errors can compound.

## Background

CLARA (Compact Linear Accelerator for Research and Applications) is a test facility at STFC's Daresbury Laboratory in Warrington, UK. It was conceived as a national platform for developing and demonstrating new accelerator, rather than as a user facility for conventional light-source science. It grew out of Daresbury's earlier accelerator R&D work, including the ALICE energy-recovery linac and the VELA (Versatile Electron Linear Accelerator) beamline, which gave industrial and academic users access to a high-quality electron beam. CLARA was designed to take that further, and its development has been led by STFC's accelerator science group (ASTeC) in partnership with the Cockcroft Institute, the joint accelerator research centre of STFC and several northern UK universities. 

The facility recently completed an upgrade to provide a 250MeV, 250pC, 100Hz beam to a versatlie user experiment area called FEBE[1] (Full Energy Beam Exploitation) which provides an interaction point for the beam and a 120TW pulsed laser. CLARA is now transitioning to a user facility, providing a unique machine for a wide range of user experiments. Target applications include laser driven plasma wakefield experiments, VHEE, and development of novel accelerator technologies.

## Motivation

Shift-start setup usually requires at least an hour of manual tweaking to reach a good state, from a loaded configuration. Developing a new setup requires even longer. 
Reducing operator overhead and eliminating this setup cost will allow for much more efficient use of beam time, improving the science/cost ratio!

## Current state / prior work

RL based steering through a section of CLARA has been attempted with some success. The RL4AA'26 workshop demonstrated this in simulation for a simple target and reduced set of magnets. 
A real deployment was attempted in April, though the model learned some non-optimal approaches and the sim2real gap presented a real challenge - particularly in beam measurement from screens.


## References

[1] Snedden, E. W., et al. "Specification and design for full energy beam exploitation of the compact linear accelerator for research and applications." Physical Review Accelerators and Beams 27.4 (2024): 041602.

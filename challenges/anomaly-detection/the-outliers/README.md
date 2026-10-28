# The Outliers

Anomaly detection challenge (ESS time series), AISSAI Hackathon 2026.

Members: Charly Lassalle (captain), Yuna Dan, Anass Bellachehab, Tudor Stefan, Stéphane Ilic, Lex Avstreikh (joker), Ishkhan Gorgisyan (case holder).

Approach: data-on-demand captures taken as nominal for training, post-mortem captures for testing, grouped by RF cavity. A Conv1D autoencoder on the six full-resolution waveforms of one cavity (MBL-090), with a reconstruction-error threshold, then latent-space clustering. Next steps: all cavities, scalar data (BCM, BPM), precursors in the 30 minutes before each fault.

Final presentation: [Indico, 16 October 2026](https://indico.in2p3.fr/event/39704/contributions/178838/).

This folder is the team's working copy on the hackathon cluster as of 16 October 2026 (code, notebooks and docs only; no data). There was no git history.

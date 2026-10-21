# Anomaly Hunters

Anomaly detection challenge (ESS time series), AISSAI Hackathon 2026.

Members: Fatima Basbous (captain), Hidekazu Masuda, Alexandros Mylonas, Lex Avstreikh, Stéphane Ilic, Ishkhan Gorgisyan (case holder).

Approach: a hybrid pipeline. An autoencoder trained on nominal pulses only, then XGBoost classifying its residual features into nominal and trip pulses, on the field plateau before the collapse. Univariate and multivariate (6 PVs) inputs compared over 5 shuffles; PCA clustering of trip waveforms with operating parameters (beam mode, beam present, trigger delay).

Final presentation: [Indico, 16 October 2026](https://indico.in2p3.fr/event/39704/contributions/178839/).

This folder is the team's working copy on the hackathon cluster as of 16 October 2026 (code, notebooks and docs only; no data). There was no git history.

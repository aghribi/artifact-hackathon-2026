"""Score an anomaly-detection submission: reconstruction MSE on the fixed holdout split.

No ground-truth fault labels exist for this dataset (see kickoff_notebook.ipynb), so there
is no precision/recall to compute. The score is model quality on unseen data: train only on
`train_idx`, report mean reconstruction error on `holdout_idx` -- lower is better.

Run as a script to reproduce the reference bar from the kickoff notebook's autoencoder:
    python score.py
Or import `score_holdout(model, cmp1)` from your own code once you have a trained model.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SAMPLE_PATH = os.path.join("data", "sample", "ess_llrf_sample.parquet")
SPLIT_PATH = os.path.join("data", "sample", "holdout_ids.json")


def load_split():
    df = pd.read_parquet(SAMPLE_PATH)
    split = json.load(open(SPLIT_PATH))
    cmp1 = np.stack(df["Q"].to_numpy()).astype(np.float32)
    return df, cmp1, np.array(split["train_idx"]), np.array(split["holdout_idx"])


class RFAutoencoder(nn.Module):
    # same architecture as kickoff_notebook.ipynb section 5
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 32), nn.Unflatten(1, (64, 32)),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(32, 1, kernel_size=8, stride=4, padding=2),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def score_holdout(model, cmp1_normalised, holdout_idx):
    """Mean per-event reconstruction MSE on the holdout split. Lower is better."""
    model.eval()
    with torch.no_grad():
        X = torch.tensor(cmp1_normalised[holdout_idx][:, None, :])
        recon = model(X)
        mse_per_event = ((X - recon) ** 2).mean(dim=(1, 2)).numpy()
    return float(mse_per_event.mean()), mse_per_event


def train_reference(cmp1, train_idx, epochs=30, seed=42):
    torch.manual_seed(seed)
    mu, sigma = cmp1[train_idx].mean(), cmp1[train_idx].std() + 1e-8
    cmp1_n = (cmp1 - mu) / sigma

    model = RFAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(cmp1_n[train_idx][:, None, :])), batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, cmp1_n


if __name__ == "__main__":
    df, cmp1, train_idx, holdout_idx = load_split()
    print(f"train: {len(train_idx)} events, holdout: {len(holdout_idx)} events")

    model, cmp1_n = train_reference(cmp1, train_idx)
    mean_mse, per_event = score_holdout(model, cmp1_n, holdout_idx)
    print(f"holdout mean reconstruction MSE: {mean_mse:.4f}")
    print(f"holdout median reconstruction MSE: {np.median(per_event):.4f}")

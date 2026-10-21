"""Convolutional autoencoder on the DD waveforms of a cavity, scored on captures it has not seen.

The question it answers: does the reconstruction error of a network trained on normal pulses say more than the
per-pulse features of ess_dd_features.py?

Input is a DD parquet (one row per pv and cycle, the waveform in data_array) and the feature table written for the
same pulses by ess_dd_subset_features.py. The three amplitude waveforms of a pulse (field, forward, reflected) are
reduced to <points> bin means and stacked as three channels. The phases are left out: they wrap at 180 degrees and
are noise outside the pulse.

The captures are split in FOLDS groups. Each group is scored by a network trained on the other groups, so every
pulse gets an error from a network that never saw its capture. Pulses more than TRIM spreads from the cavity
median on a drift feature are kept out of training and still scored.

    <out.parquet>   one row per pulse: event_id, cycle_id, cavity, fold, trained_on (used for training in the other
                    folds), recon_mse and its share per channel

    python ess_dd_conv_autoencoder.py <dd.parquet> <pulses.parquet> <out.parquet> <points> <latent> <epochs>
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from ess_dd_features import CAVITY_PV, SIGNAL, bin_means

CHANNELS = ["field", "fwd", "refl"]
FOLDS = 4
TRIM = 5
BATCH = 32
LEARNING_RATE = 1e-3


class Autoencoder(nn.Module):
    """The network of the kickoff notebook of the challenge, with three input channels and a length of <points>."""

    def __init__(self, points: int, latent: int) -> None:
        super().__init__()
        width = points // 8
        self.encoder = nn.Sequential(
            nn.Conv1d(len(CHANNELS), 32, kernel_size=8, stride=4, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * width, latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64 * width), nn.Unflatten(1, (64, width)),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(32, len(CHANNELS), kernel_size=8, stride=4, padding=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def load_pulses(path: str, points: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Keys of each pulse and its three amplitude waveforms as an array (pulse, channel, point)."""
    table = pq.ParquetFile(path).read(columns=["pv", "event_id", "cycle_id", "data_array"])
    meta = table.select(["pv", "event_id", "cycle_id"]).to_pandas()
    arrays = table.column("data_array").combine_chunks()
    offsets, values = arrays.offsets.to_numpy(), arrays.values.to_numpy()
    meta = meta.join(meta.pv.str.extract(CAVITY_PV)).dropna(subset="cavity")
    meta = meta[meta["cmp"] == "0"]
    keys, waves = [], []
    for (event_id, cycle_id, cavity), g in meta.groupby(["event_id", "cycle_id", "cavity"]):
        by_signal = {SIGNAL[d]: i for i, d in zip(g.index, g.dwn)}
        if set(by_signal) != set(CHANNELS):
            continue
        keys.append({"event_id": int(event_id), "cycle_id": int(cycle_id), "cavity": cavity})
        waves.append([bin_means(values[offsets[by_signal[c]]:offsets[by_signal[c] + 1]].astype(np.float64), points) for c in CHANNELS])
    return pd.DataFrame(keys), np.asarray(waves, dtype=np.float32)


def outside_normal(pulses: pd.DataFrame) -> pd.Series:
    """Pulses more than TRIM spreads from the median of their cavity on any drift feature."""
    drift = [c for c in pulses.columns if c.endswith("_drift")]

    def robust_z(s: pd.Series) -> pd.Series:
        spread = 1.4826 * (s - s.median()).abs().median()
        return (s - s.median()) / (spread if spread > 0 else s.std())

    return (pulses.groupby("cavity")[drift].transform(robust_z).abs() > TRIM).any(axis=1)


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit(__doc__)
    dd, features, out = sys.argv[1:4]
    points, latent, epochs = (int(a) for a in sys.argv[4:7])
    keys, x = load_pulses(dd, points)
    pulses = pd.read_parquet(features)
    keys = keys.merge(pulses.assign(outside=outside_normal(pulses))[["event_id", "cycle_id", "cavity", "outside"]],
                      on=["event_id", "cycle_id", "cavity"], how="left", validate="one_to_one")
    usable = ~keys.outside.fillna(True).to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(keys)} pulses, {keys.event_id.nunique()} captures, {int((~usable).sum())} kept out of training, "
          f"{points} points, latent {latent}, {epochs} epochs, {device}", flush=True)

    error = np.zeros((len(keys), len(CHANNELS)), dtype=np.float32)
    fold_of = np.zeros(len(keys), dtype=int)
    for fold, (train, test) in enumerate(GroupKFold(FOLDS).split(x, groups=keys.event_id)):
        train = train[usable[train]]
        # Each channel is divided by its usual peak over the training pulses, so the three weigh the same in the loss.
        scale = np.median(x[train].max(axis=2), axis=0)[None, :, None]
        torch.manual_seed(fold)
        model = Autoencoder(points, latent).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        data = torch.tensor(x[train] / scale, device=device)
        model.train()
        for epoch in range(epochs):
            for batch in data[torch.randperm(len(data), device=device)].split(BATCH):
                loss = nn.functional.mse_loss(model(batch), batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        model.eval()
        with torch.no_grad():
            held = torch.tensor(x[test] / scale, device=device)
            error[test] = ((model(held) - held) ** 2).mean(dim=2).cpu().numpy()
            seen = float(((model(data) - data) ** 2).mean())
        fold_of[test] = fold
        print(f"fold {fold}: {len(train)} training pulses, {len(test)} held out, training mse {seen:.2e}, "
              f"held-out median mse {np.median(error[test].mean(axis=1)):.2e}", flush=True)

    result = keys.drop(columns="outside").assign(fold=fold_of, trained_on=usable, recon_mse=error.mean(axis=1))
    for i, channel in enumerate(CHANNELS):
        result[f"recon_mse_{channel}"] = error[:, i]
    result.to_parquet(out, index=False)
    print(f"\nheld-out mse over all pulses: mean {result.recon_mse.mean():.2e}, median {result.recon_mse.median():.2e}")
    print(result.sort_values("recon_mse", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()

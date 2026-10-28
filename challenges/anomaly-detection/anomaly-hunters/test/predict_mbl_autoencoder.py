"""Load the MBL comparison checkpoints and reconstruct new same-length waveforms."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from compare_waveform_autoencoders import PrincipalComponentAE, WaveformAE, native_reconstruction, represent
from train_mbl_autoencoder_comparison import SmallConvAE, resize256


class MBLDetector:
    def __init__(self, directory="mbl_autoencoder_comparison", method="conv256", device=None):
        self.root = Path(directory)
        self.method = method
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision("high")
        self.config = json.loads((self.root/"configuration.json").read_text())
        self.pvs, self.modes = self.config["pvs"], self.config["modes"]
        self.threshold = json.loads((self.root/f"{method}_metrics.json").read_text())["threshold"]
        self.norms = {}
        for grid in ["native", "256"]:
            table = pd.read_csv(self.root/f"normalization_{grid}.csv", float_precision="round_trip").set_index("pv").loc[self.pvs]
            self.norms[grid] = tuple(torch.tensor(table[column].to_numpy(), dtype=torch.float32, device=self.device) for column in ["mean", "std"])
        self.model = None
        if method.startswith("pca"):
            self.grid = "256" if method.startswith("pca256_") else "native"
            self.k = int(method.split("_")[1])
            with np.load(self.root/f"pca_{self.grid}.npz") as saved:
                self.pca = {key: torch.tensor(saved[key], device=self.device) for key in ["mean", "basis", "pc_mean", "pc_std"]}
            if method.endswith("_ae"):
                self.model = PrincipalComponentAE(len(self.pvs), len(self.modes), self.k)
        elif method.startswith("conv256"):
            self.grid = "256"
            self.model = SmallConvAE(len(self.pvs), len(self.modes), conditioned=method.endswith("context"))
        else:
            self.grid = "native"
            self.model = WaveformAE(len(self.pvs), len(self.modes))
        if self.model is not None:
            saved = torch.load(self.root/f"{method}.pt", map_location=self.device, weights_only=False)
            self.model.to(self.device).load_state_dict(saved["state_dict"])
            self.model.eval()

    @torch.no_grad()
    def predict(self, waveforms, pvs, beam_modes):
        """Return full-native-grid normalized errors and physical-unit reconstructions."""
        names = pd.Series(pvs).str.replace(r"-(?:PMData|DODData)$", "", case=False, regex=True)
        pv_codes = pd.Categorical(names, categories=self.pvs).codes.astype(np.int64)
        mode_codes = pd.Categorical(beam_modes, categories=self.modes).codes.astype(np.int64)
        if (pv_codes < 0).any() or (mode_codes < 0).any():
            raise ValueError("PV or beam_mode is absent from the training categories")
        pv = torch.tensor(pv_codes, device=self.device)
        mode = torch.tensor(mode_codes, device=self.device)
        original = np.stack(waveforms).astype(np.float32, copy=False)
        if original.ndim != 2 or original.shape[1] not in [5032, 35221]:
            raise ValueError("Expected a same-length batch of 5032- or 35221-point native waveforms")
        raw = torch.tensor(original[:, None], device=self.device)
        if self.grid == "256":
            values = torch.tensor(np.stack([resize256(w) for w in original])[:, None], device=self.device)
        else:
            values = raw
        mean, std = self.norms[self.grid]
        if self.method == "conv256":
            x = (values-self.config["global_256_mean"])/self.config["global_256_std"]
            reconstructed = self.model(x, pv, mode)*self.config["global_256_std"]+self.config["global_256_mean"]
        else:
            x = (values-mean[pv, None, None])/std[pv, None, None]
            if self.method.startswith("pca"):
                length = x.shape[-1]
                if length > len(self.pca["mean"]):
                    raise ValueError("Input exceeds the fitted PCA width")
                pcs = (x[:, 0]-self.pca["mean"][:length])@self.pca["basis"][:length, :self.k]
                if self.model is not None:
                    scaled = (pcs-self.pca["pc_mean"][:self.k])/self.pca["pc_std"][:self.k]
                    pcs = self.model(scaled, pv, mode)*self.pca["pc_std"][:self.k]+self.pca["pc_mean"][:self.k]
                prediction = (pcs@self.pca["basis"][:length, :self.k].T+self.pca["mean"][:length])[:, None]
            elif self.method == "conv256_context":
                prediction = self.model(x, pv, mode)
            else:
                prediction = native_reconstruction(self.model(represent(x, self.method), pv, mode), self.method, raw.shape[-1])
            reconstructed = prediction*std[pv, None, None]+mean[pv, None, None]
        if self.grid == "256":
            reconstructed = F.interpolate(reconstructed, size=raw.shape[-1], mode="linear", align_corners=True)
        native_std = self.norms["native"][1][pv, None, None]
        score = ((raw-reconstructed)/native_std).square().mean((1, 2))
        return score.cpu().numpy(), reconstructed[:, 0].cpu().numpy()


class OfficialWaveformAdapter(nn.Module):
    """Waveform-only adapter for the unchanged official score_holdout(model, X, idx)."""
    def __init__(self, directory="mbl_autoencoder_comparison", device=None):
        super().__init__()
        root = Path(directory)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config = json.loads((root/"configuration.json").read_text())
        self.model = SmallConvAE(len(config["pvs"]), len(config["modes"]), conditioned=False).to(self.device)
        saved = torch.load(root/"conv256.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.mean, self.std = config["global_256_mean"], config["global_256_std"]
        self.reference_mean, self.reference_std = config["official_mean"], config["official_std"]

    def forward(self, x):
        physical = x.to(self.device)*self.reference_std+self.reference_mean
        unused = torch.zeros(len(x), dtype=torch.long, device=self.device)
        reconstruction = self.model((physical-self.mean)/self.std, unused, unused)*self.std+self.mean
        return ((reconstruction-self.reference_mean)/self.reference_std).cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="mbl_autoencoder_comparison")
    parser.add_argument("--method", default="conv256")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    detector = MBLDetector(args.models, args.method)
    frame = pd.read_pickle(args.input)
    lengths = frame.data_array.map(len).to_numpy()
    scores = np.full(len(frame), np.nan)
    for length in sorted(set(lengths)):
        group = np.flatnonzero(lengths == length)
        for start in range(0, len(group), 128):
            idx = group[start:start+128]
            rows = frame.iloc[idx]
            scores[idx], _ = detector.predict(rows.data_array.tolist(), rows.pv.tolist(), rows.beam_mode.tolist())
    result = frame[["cycle_id", "pv", "beam_mode"]].copy()
    result["score"] = scores
    result["threshold"] = detector.threshold
    result["anomaly_candidate"] = scores > detector.threshold
    with output.open("x") as stream:
        result.to_csv(stream, index_label="original_row")
    print(f"Saved {len(result):,} waveforms; {result.anomaly_candidate.sum():,} candidates")

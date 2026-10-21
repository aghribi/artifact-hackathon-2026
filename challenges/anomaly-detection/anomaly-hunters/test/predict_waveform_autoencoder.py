"""Score waveform DataFrames using a saved conditional autoencoder comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare_waveform_autoencoders import (WaveformAE, PrincipalComponentAE, represent,
                                         native_reconstruction, waveform_error)


class WaveformDetector:
    def __init__(self, directory, method="raw", device=None):
        self.root, self.method = Path(directory), method
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        config = json.loads((self.root / "configuration.json").read_text())
        self.pvs, self.modes = config["pvs"], config["modes"]
        self.pv_codes = {v:i for i,v in enumerate(self.pvs)}
        self.mode_codes = {v:i for i,v in enumerate(self.modes)}
        norm = pd.read_csv(self.root / "normalization.csv", float_precision="round_trip").set_index("pv").loc[self.pvs]
        self.mean = torch.tensor(norm["mean"].to_numpy(),dtype=torch.float32,device=self.device)
        self.std = torch.tensor(norm["std"].to_numpy(),dtype=torch.float32,device=self.device)
        metrics = json.loads((self.root / (method + "_metrics.json")).read_text())
        self.threshold = metrics["threshold"]
        self.dimensions = int(method.split("_")[0][3:]) if method.startswith("pca") else None
        self.model = None
        if self.dimensions:
            with np.load(self.root / "pca.npz") as pca:
                self.basis = torch.tensor(pca["basis"][:,:self.dimensions],device=self.device)
                self.pca_mean = torch.tensor(pca["mean"],device=self.device)
            if method.endswith("_ae"):
                self.model = PrincipalComponentAE(len(self.pvs),len(self.modes),self.dimensions).to(self.device)
                scale_path = self.root / "pc_scaling.npz"
                if scale_path.exists():
                    with np.load(scale_path) as scale:
                        self.pc_mean = torch.tensor(scale["mean"][:self.dimensions],device=self.device)
                        self.pc_std = torch.tensor(scale["std"][:self.dimensions],device=self.device)
                else:
                    # Reproduce the original training-only PC scaling when loading an older run.
                    pcs = torch.tensor(np.load(self.root/"principal_components.npy")[:,:self.dimensions],device=self.device)
                    split = pd.read_csv(self.root/"sample_splits.csv",usecols=["split","length"])
                    train = np.flatnonzero(split.split.eq("train") & split.length.gt(0))
                    self.pc_mean = pcs[train].mean(0)
                    self.pc_std = pcs[train].std(0,correction=0).clamp(min=1e-6)
                    del pcs
        else:
            self.model = WaveformAE(len(self.pvs),len(self.modes),channels=2 if method=="minmax16" else 1).to(self.device)
        if self.model is not None:
            saved = torch.load(self.root/(method+".pt"),map_location=self.device,weights_only=False)
            self.model.load_state_dict(saved["state_dict"])
            self.model.eval()

    @torch.no_grad()
    def predict(self, waves, pvs, modes):
        """Score a same-length batch; return original-unit reconstruction and anomaly scores."""
        normalized = pd.Series(pvs).str.replace(r"-(?:DODData|PMData)$","",case=False,regex=True)
        unknown = set(normalized) - set(self.pv_codes)
        unknown_modes = set(modes) - set(self.mode_codes)
        if unknown or unknown_modes:
            raise ValueError(f"Unseen categories: PV={sorted(unknown)}, beam_mode={sorted(unknown_modes)}")
        pv = torch.tensor([self.pv_codes[p] for p in normalized],device=self.device)
        mode = torch.tensor([self.mode_codes[m] for m in modes],device=self.device)
        x = torch.from_numpy(np.stack(waves).astype(np.float32,copy=False)).to(self.device)
        x = ((x-self.mean[pv,None])/self.std[pv,None])[:,None,:]
        if self.dimensions:
            length = x.shape[-1]
            if length > len(self.pca_mean):
                raise ValueError("Waveform exceeds the fitted PCA width")
            pcs = (x[:,0]-self.pca_mean[:length])@self.basis[:length]
            if self.model is not None:
                scaled = (pcs-self.pc_mean)/self.pc_std
                with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
                    pcs = self.model(scaled,pv,mode).float()
                pcs = pcs*self.pc_std+self.pc_mean
            reconstructed = (pcs@self.basis[:length].T+self.pca_mean[:length])[:,None]
            scores = waveform_error(x,reconstructed,"raw")
        else:
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
                reconstructed = self.model(represent(x,self.method),pv,mode).float()
            reconstructed = native_reconstruction(reconstructed,self.method,x.shape[-1])
            scores = waveform_error(x,reconstructed,self.method)
        original_units = reconstructed*self.std[pv,None,None]+self.mean[pv,None,None]
        return scores.cpu().numpy(), original_units.cpu().numpy()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models",default="waveform_autoencoder_comparison")
    p.add_argument("--method",default="raw")
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    detector=WaveformDetector(args.models,args.method)
    frame=pd.read_pickle(args.input)
    results=frame[[c for c in ["cycle_id","cycle_ts","pv","beam_mode"] if c in frame]].copy()
    scores=np.full(len(frame),np.nan)
    lengths=np.array([0 if a is None else len(a) for a in frame.data_array])
    for length in sorted(set(lengths)-{0}):
        group=np.flatnonzero(lengths==length)
        for start in range(0,len(group),128):
            idx=group[start:start+128];batch=frame.iloc[idx]
            scores[idx],_=detector.predict(batch.data_array.tolist(),batch.pv.tolist(),batch.beam_mode.tolist())
    results["score"]=scores;results["threshold"]=detector.threshold
    results["label"]=pd.array(np.where(np.isfinite(scores),np.where(scores>detector.threshold,-1,1),None),dtype="Int64")
    results.to_csv(output,index=True,index_label="source_index")
    print(f"Saved {len(results):,} scores; {results.label.eq(-1).sum():,} candidates to {output}")

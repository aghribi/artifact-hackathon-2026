"""Compare waveform, downsampled and PCA autoencoders with identical held-out groups."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from cycle_anomaly_diagnosis import CycleDiagnoser

HERE = Path(__file__).resolve().parent
REPRESENTATIONS = ["raw", "decimate4", "mean4", "mean16", "minmax16"]
PCA_DIMS = [32, 64, 128]


class Conditions(nn.Module):
    def __init__(self, pvs, modes):
        super().__init__()
        self.pv = nn.Embedding(pvs, 16)
        self.mode = nn.Embedding(modes, 4)

    def forward(self, pv, mode):
        return torch.cat([self.pv(pv), self.mode(mode)], dim=1)


class WaveformAE(nn.Module):
    """Native length input/output; only internal features are pooled/interpolated."""
    def __init__(self, pvs, modes, channels=1, latent=32):
        super().__init__()
        self.condition = Conditions(pvs, modes)
        widths = [channels, 16, 32, 32]
        specs = [(8, 4, 2), (4, 2, 1), (4, 2, 1)]
        self.encoder = nn.ModuleList([nn.Conv1d(a, b, k, s, p)
            for (a, b), (k, s, p) in zip(zip(widths[:-1], widths[1:]), specs)])
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.to_latent = nn.Linear(32 * 32 + 20, latent)
        self.from_latent = nn.Linear(latent + 20, 32 * 32)
        self.decoder = nn.ModuleList([nn.ConvTranspose1d(b, a, k, s, p)
            for (a, b), (k, s, p) in reversed(list(zip(zip(widths[:-1], widths[1:]), specs)))])

    def forward(self, x, pv, mode):
        condition = self.condition(pv, mode)
        sizes = []
        for layer in self.encoder:
            sizes.append(x.shape)
            x = F.gelu(layer(x))
        encoded_length = x.shape[-1]
        z = self.to_latent(torch.cat([self.pool(x).flatten(1), condition], dim=1))
        x = self.from_latent(torch.cat([z, condition], dim=1)).reshape(-1, 32, 32)
        x = F.interpolate(x, size=encoded_length, mode="linear", align_corners=False)
        for j, (layer, shape) in enumerate(zip(self.decoder, reversed(sizes))):
            x = layer(x, output_size=shape)
            if j < len(self.decoder) - 1:
                x = F.gelu(x)
        return x


class PrincipalComponentAE(nn.Module):
    def __init__(self, pvs, modes, components):
        super().__init__()
        self.condition = Conditions(pvs, modes)
        self.encoder = nn.Sequential(nn.Linear(components + 20, 128), nn.GELU(),
                                     nn.Linear(128, max(8, components // 4)))
        self.decoder = nn.Sequential(nn.Linear(max(8, components // 4) + 20, 128), nn.GELU(),
                                     nn.Linear(128, components))

    def forward(self, x, pv, mode):
        c = self.condition(pv, mode)
        z = self.encoder(torch.cat([x, c], 1))
        return self.decoder(torch.cat([z, c], 1))


def represent(x, name):
    if name == "raw":
        return x
    if name == "decimate4":
        return x[..., ::4]
    if name.startswith("mean"):
        factor = int(name[4:])
        return F.avg_pool1d(x, factor, factor, ceil_mode=True, count_include_pad=False)
    if name == "minmax16":
        return torch.cat([-F.max_pool1d(-x, 16, 16, ceil_mode=True),
                          F.max_pool1d(x, 16, 16, ceil_mode=True)], 1)
    raise ValueError(name)


def native_reconstruction(reconstructed, name, length):
    if name == "raw":
        return reconstructed
    if name.startswith("mean"):
        return reconstructed.repeat_interleave(int(name[4:]), dim=-1)[..., :length]
    if name == "decimate4":
        positions = torch.arange(length, device=reconstructed.device)
        left = (positions // 4).clamp(max=reconstructed.shape[-1] - 1)
        right = (left + 1).clamp(max=reconstructed.shape[-1] - 1)
        weight = (positions % 4).float() / 4
        return reconstructed[..., left] * (1-weight) + reconstructed[..., right] * weight
    if name == "minmax16":
        return reconstructed.repeat_interleave(16, dim=-1)[..., :length]
    raise ValueError(name)


def waveform_error(x, reconstructed, name):
    if name == "minmax16":
        lo, hi = torch.minimum(reconstructed[:, :1], reconstructed[:, 1:]), torch.maximum(reconstructed[:, :1], reconstructed[:, 1:])
        return (F.relu(lo - x).square() + F.relu(x - hi).square()).mean((1, 2))
    return (x - reconstructed).square().mean((1, 2))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class Experiment:
    def __init__(self, args):
        self.args, self.output = args, Path(args.output).resolve()
        self.output.mkdir(exist_ok=True)
        self.device = torch.device(args.device)
        torch.set_num_threads(args.cpu_threads)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.started = time.monotonic()
        self.log("Loading the filtered training dataset")
        filtered = pd.read_pickle(args.input)
        excluded = pd.read_csv(HERE / "waveform_anomaly_comparison/global_pca_knn_sigma5_cycles.csv").cycle_id
        if filtered.cycle_id.isin(excluded).any():
            raise ValueError("Excluded candidate cycles remain in training input")
        self.extra_anomaly_cycles = set()
        if args.anomaly_cycles:
            labels = pd.read_csv(args.anomaly_cycles)
            if "cycle_id" not in labels or labels.cycle_id.isna().any():
                raise ValueError("Anomaly cycle CSV must contain non-null cycle_id values")
            self.extra_anomaly_cycles = set(map(int, labels.cycle_id))
            filtered = filtered.loc[~filtered.cycle_id.isin(self.extra_anomaly_cycles)]
            excluded = pd.Series(sorted(set(map(int, excluded)) | self.extra_anomaly_cycles))
        self.filtered_index = filtered.index.to_numpy()
        self.log("Loading excluded-cycle waveforms for evaluation only")
        original = pd.read_pickle(args.original)
        missing = self.extra_anomaly_cycles - set(map(int, original.cycle_id.unique()))
        if missing:
            raise ValueError(f"Labelled anomalous cycles are absent from --original: {sorted(missing)}")
        extra = original.loc[original.cycle_id.isin(excluded)]
        frame = pd.concat([filtered, extra]).sort_index()
        assert frame.index.is_unique and np.array_equal(frame.index, original.index)
        assert np.array_equal(frame.cycle_id, original.cycle_id) and np.array_equal(frame.pv, original.pv)
        del original, filtered, extra
        self.arrays = frame.data_array.to_numpy()
        self.meta = frame[["cycle_id", "cycle_ts", "pv", "beam_mode", "source_archive"]].copy()
        self.meta.insert(0, "original_row", frame.index.to_numpy())
        self.meta["source_pv"] = self.meta.pv
        self.meta.pv = self.meta.pv.str.replace(r"-(?:DODData|PMData)$", "", case=False, regex=True)
        self.meta["length"] = [0 if x is None else len(x) for x in self.arrays]
        del frame
        self.meta["split"] = "excluded_evaluation"
        self.meta.loc[self.meta.original_row.isin(self.filtered_index), "split"] = "unassigned"
        self.make_splits()
        self.pvs = sorted(self.meta.pv.unique())
        self.modes = sorted(self.meta.beam_mode.unique())
        self.pv = torch.tensor(pd.Categorical(self.meta.pv, categories=self.pvs).codes.astype(np.int64), device=self.device)
        self.mode = torch.tensor(pd.Categorical(self.meta.beam_mode, categories=self.modes).codes.astype(np.int64), device=self.device)
        self.lengths = self.meta.length.to_numpy()
        self.valid = self.lengths > 0
        self.splits = {s: np.flatnonzero(self.meta.split.eq(s) & self.valid) for s in ["train", "validation", "calibration", "test", "excluded_evaluation"]}
        assert set(self.meta.iloc[self.splits["train"]].pv) == set(self.pvs)
        assert set(self.meta.iloc[self.splits["train"]].beam_mode) == set(self.modes)
        self.known = pd.read_csv(HERE / "waveform_anomaly_comparison/known_anomalies.csv")
        self.known_positions = self.known.original_row.to_numpy(dtype=int)
        assert self.meta.iloc[self.known_positions].split.eq("excluded_evaluation").all()
        self.confirmed_cycle_ids = set(map(int, self.known.cycle_id)) | self.extra_anomaly_cycles
        split_file = self.output / "sample_splits.csv"
        if split_file.exists() and list(self.output.glob("*.pt")):
            columns = ["original_row", "pv", "cycle_id", "beam_mode", "length", "split"]
            previous_splits = pd.read_csv(split_file, usecols=columns)
            try:
                pd.testing.assert_frame_equal(previous_splits[columns], self.meta[columns].reset_index(drop=True), check_dtype=False)
            except AssertionError as error:
                raise ValueError("Data or splits changed; use a new --output directory instead of reusing old checkpoints") from error
        self.fit_normalization()
        self.meta.to_csv(self.output / "sample_splits.csv", index=False)
        save_json(self.output / "configuration.json", dict(**vars(args), pvs=self.pvs, modes=self.modes,
            normalization="Fixed scalar mean/std per PV, fitted on train samples only; never per waveform",
            split="Archives grouped and stratified by beam mode; no cycle appears in multiple partitions",
            counts={s:len(v) for s,v in self.splits.items()},
            training_input=str(Path(args.input).resolve()), empty_waveforms=int((~self.valid).sum()),
            scoring="Original waveform MSE in fixed per-PV training units; minmax uses distance outside its reconstructed envelope",
            objective="Notebook-style normal-waveform reconstruction AE; anomaly score is reconstruction error",
            cycle_decision="A cycle is an anomaly candidate iff at least one PV waveform score exceeds the unchanged AE threshold",
            confirmed_anomaly_cycles=sorted(self.confirmed_cycle_ids),
            threshold="Median + 5 * population std of calibration errors only", torch=torch.__version__))
        self.results = []

    def log(self, message):
        line = f"[{time.monotonic() - self.started:.1f}s] {message}"
        print(line, flush=True)
        with (self.output / "run.log").open("a") as stream:
            stream.write(line + "\n")

    def make_splits(self):
        normal = self.meta.loc[self.meta.split.eq("unassigned")]
        # Connect archives if they share any cycle, avoiding capture leakage.
        parents = {a:a for a in normal.source_archive.unique()}
        def root(a):
            while parents[a] != a:
                parents[a] = parents[parents[a]]
                a = parents[a]
            return a
        for _, group in normal.groupby("cycle_id", sort=False):
            archives = group.source_archive.unique()
            for a in archives[1:]:
                parents[root(a)] = root(archives[0])
        normal = normal.assign(group=normal.source_archive.map(root))
        group_modes = normal.groupby("group").beam_mode.agg(lambda v:"|".join(sorted(v.unique())))
        assignment = {}
        for _, categories in group_modes.groupby(group_modes):
            groups = sorted(categories.index)
            self.rng.shuffle(groups)
            if len(groups) < 4:
                raise ValueError("Too few independent capture groups for a mode-stratified four-way split")
            size = max(1, round(len(groups) * .1))
            for split, members in [("test", groups[:size]), ("calibration", groups[size:2*size]),
                ("validation", groups[2*size:3*size]), ("train", groups[3*size:])]:
                assignment.update({g:split for g in members})
        self.meta.loc[normal.index, "split"] = normal.group.map(assignment)
        assert self.meta.groupby("cycle_id").split.nunique().max() == 1
        for split in ["train", "validation", "calibration", "test"]:
            subset = self.meta.loc[self.meta.split.eq(split)]
            self.log(f"{split}: {len(subset):,} rows / {subset.cycle_id.nunique()} cycles / {subset.source_archive.nunique()} archives")

    def fit_normalization(self):
        self.log("Fitting training-only PV normalization")
        codes = self.pv.cpu().numpy()
        sums, squares, counts = (np.zeros(len(self.pvs), dtype=np.float64) for _ in range(3))
        for j in self.splits["train"]:
            x = self.arrays[j]
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite raw waveform at original_row={j}")
            p = codes[j]
            sums[p] += np.sum(x, dtype=np.float64)
            squares[p] += np.sum(np.square(x, dtype=np.float64), dtype=np.float64)
            counts[p] += len(x)
        means = sums / counts
        stds = np.sqrt(np.maximum(squares / counts - means**2, 0))
        stds[stds < 1e-6] = 1
        self.mean = torch.tensor(means, dtype=torch.float32, device=self.device)
        self.std = torch.tensor(stds, dtype=torch.float32, device=self.device)
        pd.DataFrame(dict(pv=self.pvs,mean=means,std=stds,training_samples=counts)).to_csv(self.output / "normalization.csv", index=False)

    def batches(self, indices, shuffle=False, batch_size=None):
        batches = []
        for length in sorted(set(self.lengths[indices])):
            selected = indices[self.lengths[indices] == length].copy()
            if shuffle:
                self.rng.shuffle(selected)
            size = batch_size or self.args.batch_size
            batches += [selected[k:k+size] for k in range(0, len(selected), size)]
        if shuffle:
            self.rng.shuffle(batches)
        return batches

    def wave_batch(self, idx):
        x = torch.from_numpy(np.stack(self.arrays[idx])).to(self.device)
        return ((x - self.mean[self.pv[idx], None]) / self.std[self.pv[idx], None])[:, None, :]

    def train_model(self, name, model, features=None):
        path = self.output / (name + ".pt")
        history_path = self.output / (name + "_history.csv")
        model = model.to(self.device)
        if path.exists() and history_path.exists():
            saved = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(saved["state_dict"])
            self.log(f"Reusing completed {name}")
            return model, saved["training_seconds"], pd.read_csv(history_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        best, best_state, stale, history = float("inf"), None, 0, []
        started = time.monotonic()
        checkpoint_path = self.output / (name + "_checkpoint.pt")
        first_epoch = 1
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "current_state" in checkpoint:
                model.load_state_dict(checkpoint["current_state"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                best_state, best = checkpoint["state_dict"], checkpoint["best"]
                stale, history = checkpoint["stale"], checkpoint["history"]
                first_epoch = checkpoint["epoch"] + 1
                started -= checkpoint["training_seconds"]
                self.rng.bit_generator.state = checkpoint["rng_state"]
                self.log(f"Resuming {name} at epoch {first_epoch}")
        train, validation = self.splits["train"], self.splits["validation"]
        epoch = first_epoch - 1
        for epoch in range(first_epoch, self.args.epochs + 1):
            record = dict(epoch=epoch)
            for split, indices in [("train",train),("validation",validation)]:
                model.train(split == "train")
                total, count = 0., 0
                size = 2048 if features is not None else None
                batches = self.batches(indices, split=="train", size)
                last_update = time.monotonic()
                with torch.set_grad_enabled(split == "train"):
                    for batch_number, idx in enumerate(batches, 1):
                        target = features[idx] if features is not None else represent(self.wave_batch(idx), name)
                        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type=="cuda"):
                            recon = model(target, self.pv[idx], self.mode[idx])
                        loss = (recon.float() - target.float()).square().flatten(1).mean(1).mean()
                        if not torch.isfinite(loss):
                            raise ValueError(f"Nonfinite loss in {name}, {split}, epoch {epoch}")
                        if split == "train":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                        total += float(loss.detach()) * len(idx)
                        count += len(idx)
                        if time.monotonic() - last_update > 20:
                            self.log(f"{name} epoch {epoch} {split}: batch {batch_number}/{len(batches)}")
                            save_json(self.output / "progress.json", dict(method=name,epoch=epoch,phase=split,
                                batch=batch_number,batches=len(batches),completed_methods=len(self.results)))
                            last_update = time.monotonic()
                record[split + "_loss"] = total / count
            history.append(record)
            self.log(f"{name} epoch {epoch}: train={record['train_loss']:.6g}, validation={record['validation_loss']:.6g}")
            pd.DataFrame(history).to_csv(history_path, index=False)
            if record["validation_loss"] < best * (1 - 1e-4):
                best = record["validation_loss"]
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            torch.save(dict(state_dict=best_state, training_seconds=time.monotonic()-started,
                current_state=model.state_dict(),optimizer=optimizer.state_dict(),best=best,stale=stale,
                history=history,rng_state=self.rng.bit_generator.state,
                epoch=epoch, finished=False), checkpoint_path)
            if stale >= self.args.patience:
                break
        model.load_state_dict(best_state)
        duration = time.monotonic()-started
        torch.save(dict(state_dict=best_state, training_seconds=duration, epoch=epoch, finished=True), path)
        return model, duration, pd.DataFrame(history)

    def score_model(self, name, model, duration, history, pca_dim=None, features=None):
        path = self.output / (name + "_scores.npy")
        if path.exists():
            scores = np.load(path)
        else:
            scores = np.full(len(self.meta), np.nan)
            model.eval() if model is not None else None
            started = time.monotonic()
            previews = {}
            known = set(self.known_positions)
            with torch.no_grad():
                for idx in self.batches(np.flatnonzero(self.valid)):
                    x = self.wave_batch(idx)
                    if pca_dim is not None:
                        if model is None:
                            pcs = self.pcs[idx,:pca_dim]
                        else:
                            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
                                pcs = model(features[idx],self.pv[idx],self.mode[idx]).float()
                            pcs = pcs * self.pc_std[:pca_dim] + self.pc_mean[:pca_dim]
                        reconstructed = (pcs @ self.basis[:x.shape[-1],:pca_dim].T + self.pca_mean[:x.shape[-1]])[:,None,:]
                        error = waveform_error(x, reconstructed, "raw")
                    else:
                        target = represent(x,name)
                        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
                            reconstructed = model(target,self.pv[idx],self.mode[idx]).float()
                        reconstructed = native_reconstruction(reconstructed,name,x.shape[-1])
                        error = waveform_error(x,reconstructed,name)
                    scores[idx] = error.cpu().numpy()
                    for k,j in enumerate(idx):
                        if j in known:
                            scale = self.std[self.pv[j]].item();offset=self.mean[self.pv[j]].item()
                            previews[str(j)] = (reconstructed[k].cpu().numpy()*scale+offset)
            np.save(path,scores)
            np.savez(self.output / (name + "_known_reconstructions.npz"),**previews)
            self.log(f"Scored {name} in {time.monotonic()-started:.1f}s")
        calibration = scores[self.splits["calibration"]]
        median, sigma = float(np.median(calibration)), float(np.std(calibration,ddof=0))
        cutoff = median + 5*sigma
        candidate = scores > cutoff
        test = self.splits["test"]
        result = dict(method=name,epochs=0 if history is None else len(history),training_seconds=duration,
            epoch_limit_reached=False if history is None else len(history)>=self.args.epochs,
            best_validation_loss=None if history is None else float(history.validation_loss.min()),
            calibration_median=median,calibration_std=sigma,threshold=cutoff,
            test_waveforms=len(test),test_candidates=int(candidate[test].sum()),
            test_mean_waveform_score=float(np.mean(scores[test])),
            test_candidate_rate=float(candidate[test].mean()),all_candidates=int(candidate.sum()),
            candidate_cycles=int(self.meta.loc[candidate,'cycle_id'].nunique()),
            known_detected=int(candidate[self.known_positions].sum()),known_total=len(self.known_positions),
            parameters=0 if model is None else sum(p.numel() for p in model.parameters()),
            score_definition="Mean squared distance outside reconstructed min/max envelope" if name=="minmax16" else "Full original-length normalized waveform MSE",
            pca_explained_variance=None if pca_dim is None else float(self.pca_variance[:pca_dim].sum()))
        waves, cycles = CycleDiagnoser(cutoff).predict(self.meta.assign(score=scores))
        cycles.to_csv(self.output / (name + "_cycle_predictions.csv"), index=False)
        waves.loc[waves.pv_anomaly_candidate].to_csv(self.output / (name + "_candidate_pvs.csv"), index=False)
        assert int(cycles.anomaly_candidate.eq(True).sum()) == result["candidate_cycles"]
        known_cycles = cycles.loc[cycles.cycle_id.isin(self.confirmed_cycle_ids)]
        test_cycles = cycles.loc[cycles["split"].eq("test")]
        result.update(known_cycles_detected=int(known_cycles.anomaly_candidate.eq(True).sum()),
            known_cycles_total=len(known_cycles), test_cycles=len(test_cycles),
            test_candidate_cycles=int(test_cycles.anomaly_candidate.eq(True).sum()),
            test_cycle_candidate_rate=float(test_cycles.anomaly_candidate.eq(True).mean()))
        self.results.append(result)
        save_json(self.output / (name + "_metrics.json"),result)
        self.publish()
        if model is not None:
            del model
        torch.cuda.empty_cache() if self.device.type=="cuda" else None

    def fit_pca(self):
        cache = self.output / "pca.npz"
        if cache.exists():
            with np.load(cache) as z:
                self.basis=torch.tensor(z['basis'],device=self.device)
                self.pca_mean=torch.tensor(z['mean'],device=self.device)
                self.pca_variance=z['explained_variance']
            self.pcs=torch.tensor(np.load(self.output/'principal_components.npy'),device=self.device)
        else:
            self.log("Fitting shared PCA using training waveforms only, no resampling")
            started=time.monotonic()
            maximum=int(self.lengths.max());train=self.splits['train']
            groups=[train[self.lengths[train]==l] for l in sorted(set(self.lengths[train]))]
            order=np.concatenate(groups)
            matrix=torch.zeros((len(train),maximum),device=self.device)
            counts=torch.zeros(maximum,device=self.device)
            offset=0
            for group in groups:
                length=int(self.lengths[group[0]]);counts[:length]+=len(group)
                for start in range(0,len(group),256):
                    idx=group[start:start+256];x=self.wave_batch(idx)[:,0]
                    matrix[offset+start:offset+start+len(idx),:length]=x
                offset+=len(group)
            self.pca_mean=matrix.sum(0)/counts.clamp(min=1)
            offset=0
            for group in groups:
                matrix[offset:offset+len(group),:self.lengths[group[0]]].sub_(self.pca_mean[:self.lengths[group[0]]])
                offset+=len(group)
            energy=matrix.square().sum().item()
            torch.manual_seed(self.args.seed)
            u,s,v=torch.pca_lowrank(matrix,q=144,center=False,niter=2)
            self.basis=v[:,:128].contiguous()
            self.pca_variance=(s[:128].square()/energy).cpu().numpy()
            del matrix,u,s,v
            torch.cuda.empty_cache()
            self.pcs=torch.zeros((len(self.meta),128),device=self.device)
            with torch.no_grad():
                for idx in self.batches(np.flatnonzero(self.valid)):
                    x=self.wave_batch(idx)[:,0];length=x.shape[-1]
                    self.pcs[idx]=(x-self.pca_mean[:length])@self.basis[:length]
            np.savez(cache,basis=self.basis.cpu().numpy(),mean=self.pca_mean.cpu().numpy(),explained_variance=self.pca_variance)
            np.save(self.output/'principal_components.npy',self.pcs.cpu().numpy())
            self.log(f"PCA fitted and projected in {time.monotonic()-started:.1f}s; 128 components retain {self.pca_variance.sum():.2%}")
        self.pc_mean=self.pcs[self.splits['train']].mean(0)
        self.pc_std=self.pcs[self.splits['train']].std(0,correction=0).clamp(min=1e-6)

    def publish(self):
        pd.DataFrame(self.results).to_csv(self.output/'comparison.csv',index=False)
        save_json(self.output/'summary.json',dict(results=self.results,
            note="Normal candidates selected upstream by PCA+kNN, not confirmed normals. Test candidate rate is not a validated false-positive rate. Known positives are evaluation examples, not a complete accuracy estimate."))
        rows=[]
        for result in self.results:
            scores=np.load(self.output/(result['method']+'_scores.npy'))
            for j in self.known_positions:
                rows.append(dict(method=result['method'],original_row=int(j),pv=self.meta.pv.iloc[j],
                    cycle_id=int(self.meta.cycle_id.iloc[j]),beam_mode=self.meta.beam_mode.iloc[j],
                    score=float(scores[j]),threshold=result['threshold'],detected=bool(scores[j]>result['threshold'])))
        pd.DataFrame(rows).to_csv(self.output/'known_anomaly_results.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        names=[r['method'] for r in self.results]
        axes[0].bar(names,[r['test_candidate_rate']*100 for r in self.results],color='#2563eb')
        axes[0].set_ylabel('Unlabelled held-out candidate rate (%)');axes[0].set_title('Median + 5 std of calibration errors')
        axes[1].bar(names,[r['training_seconds']/60 for r in self.results],color='#15966b')
        axes[1].set_ylabel('AE training time (min; PCA fit excluded)');axes[1].set_title('Training cost')
        for ax in axes:ax.tick_params(axis='x',rotation=55)
        fig.savefig(self.output/'comparison.png',dpi=150);plt.close(fig)

    def run(self):
        for name in REPRESENTATIONS:
            self.log('Starting '+name)
            torch.manual_seed(self.args.seed)
            model=WaveformAE(len(self.pvs),len(self.modes),channels=2 if name=='minmax16' else 1)
            model,duration,history=self.train_model(name,model)
            self.score_model(name,model,duration,history)
            del model
        self.fit_pca()
        for dimensions in PCA_DIMS:
            self.score_model(f'pca{dimensions}',None,0,None,pca_dim=dimensions)
            name=f'pca{dimensions}_ae'
            torch.manual_seed(self.args.seed)
            features=(self.pcs[:,:dimensions]-self.pc_mean[:dimensions])/self.pc_std[:dimensions]
            model=PrincipalComponentAE(len(self.pvs),len(self.modes),dimensions)
            model,duration,history=self.train_model(name,model,features)
            self.score_model(name,model,duration,history,pca_dim=dimensions,features=features)
            del model,features
        predictions=self.meta.copy()
        for result in self.results:
            scores=np.load(self.output/(result['method']+'_scores.npy'))
            predictions[result['method']+'_score']=scores
            predictions[result['method']+'_label']=pd.array(np.where(np.isfinite(scores),np.where(scores>result['threshold'],-1,1),None),dtype='Int64')
        predictions.to_csv(self.output/'predictions.csv',index=False)
        for result in self.results:
            name=result['method'];selected=predictions.loc[predictions[name+'_label'].eq(-1)]
            selected[['original_row','pv','cycle_id','cycle_ts','beam_mode','split',name+'_score']].to_csv(self.output/(name+'_candidates.csv'),index=False)
        self.log('All comparisons complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default=str(HERE/'dd_all_cycles_without_global_pca_knn_5sigma_cycles.pkl'))
    p.add_argument('--original',default=str(HERE/'dd_all_cycles.pkl'))
    p.add_argument('--output',default=str(HERE/'waveform_autoencoder_comparison'))
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cpu-threads',type=int,default=8)
    p.add_argument('--anomaly-cycles',default=None,help='Confirmed anomalous cycle_id CSV; exclude all its PVs from normal-data partitions and evaluate cycles, without assigning fault labels to individual PVs')
    Experiment(p.parse_args()).run()

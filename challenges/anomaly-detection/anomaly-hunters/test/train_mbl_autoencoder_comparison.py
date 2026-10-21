"""Train MBL Dwn0-Cmp1 waveform AEs with common, cycle-disjoint partitions.

Official holdout cycles never enter training, validation, calibration or PCA fitting.
Native-grid reconstruction errors and official 256-point MSE are reported separately.
"""
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

from compare_waveform_autoencoders import Conditions, PrincipalComponentAE, WaveformAE, native_reconstruction, represent
from cycle_anomaly_diagnosis import CycleDiagnoser

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def resize256(waveform):
    """Same full-span linear interpolation as the distributed sample (then float32)."""
    waveform = np.asarray(waveform)
    coordinates = np.linspace(0, len(waveform)-1, 256)
    return np.interp(coordinates, np.arange(len(waveform)), waveform).astype(np.float32)


class SmallConvAE(nn.Module):
    def __init__(self, pvs, modes, conditioned=False, latent=16):
        super().__init__()
        self.condition = Conditions(pvs, modes) if conditioned else None
        context = 20 if conditioned else 0
        self.encoder = nn.Sequential(nn.Conv1d(1, 16, 8, 4, 2), nn.GELU(),
                                     nn.Conv1d(16, 32, 4, 2, 1), nn.GELU(), nn.Flatten())
        self.to_latent = nn.Linear(32*32+context, latent)
        self.from_latent = nn.Linear(latent+context, 32*32)
        self.decoder = nn.Sequential(nn.Unflatten(1, (32, 32)),
                                     nn.ConvTranspose1d(32, 16, 4, 2, 1), nn.GELU(),
                                     nn.ConvTranspose1d(16, 1, 8, 4, 2))

    def forward(self, x, pv, mode):
        features = self.encoder(x)
        context = self.condition(pv, mode) if self.condition is not None else None
        if context is not None:
            features = torch.cat((features, context), 1)
        latent = self.to_latent(features)
        if context is not None:
            latent = torch.cat((latent, context), 1)
        return self.decoder(self.from_latent(latent))


def grouped_splits(meta, seed):
    """Keep archives connected by shared cycle IDs in the same partition."""
    parents = {str(a): str(a) for a in meta.source_archive.unique()}
    def root(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a
    for _, rows in meta.groupby("cycle_id", sort=False):
        archives = rows.source_archive.unique()
        for archive in archives[1:]:
            parents[root(archive)] = root(archives[0])
    groups = meta.source_archive.map(root)
    group_modes = meta.assign(group=groups).groupby("group").beam_mode.agg(lambda s: "|".join(sorted(s.unique())))
    rng = np.random.default_rng(seed)
    assignment = {}
    for _, categories in group_modes.groupby(group_modes):
        keys = np.array(sorted(categories.index), dtype=object)
        rng.shuffle(keys)
        if len(keys) < 4:
            raise ValueError("At least four independent capture groups per mode are required")
        size = max(1, round(len(keys)*.1))
        for split, members in [("test", keys[:size]), ("calibration", keys[size:2*size]),
                               ("validation", keys[2*size:3*size]), ("train", keys[3*size:])]:
            assignment.update({key: split for key in members})
    return groups.map(assignment)


class Comparison:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/"learning_curves").mkdir(exist_ok=True)
        self.started = time.monotonic()
        self.device = torch.device(args.device)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.rng = np.random.default_rng(args.seed)
        self.results = []
        self.official_results = []
        self.pcas = {}
        self.log("Reading prepared MBL normal candidates")
        frame = pd.read_pickle(args.input)
        sample_root = HERE/"official_anomaly_reference/data/sample"
        self.sample = pd.read_parquet(sample_root/"ess_llrf_sample.parquet")
        split = json.loads((sample_root/"holdout_ids.json").read_text())
        self.official_idx = np.asarray(split["holdout_idx"], dtype=int)
        self.official_cycles = set(map(int, self.sample.iloc[self.official_idx].cycle_id))
        if frame.cycle_id.isin(self.official_cycles).any():
            raise ValueError("Official evaluation cycles are present in the input")
        frame = frame.copy()
        source_rows = frame.index.to_numpy()
        frame["source_pv"] = frame.pv
        frame["pv"] = frame.pv.str.replace(r"-(?:PMData|DODData)$", "", case=False, regex=True)
        if not frame.pv.str.fullmatch(r"MBL.*:Dwn0-Cmp1SDS").all():
            raise ValueError("Input includes PVs outside MBL Dwn0-Cmp1")
        arrays = frame.data_array.to_numpy()
        lengths = np.array([len(w) for w in arrays], dtype=int)
        if not np.isin(lengths, [5032, 35221]).all():
            raise ValueError("Unexpected or empty waveform length")
        self.meta = frame[["cycle_id", "cycle_ts", "pv", "source_pv", "beam_mode", "source_archive"]].reset_index(drop=True)
        self.meta.insert(0, "original_row", source_rows)
        self.meta["length"] = lengths
        self.meta["split"] = grouped_splits(self.meta, args.seed)
        self.pvs = sorted(self.meta.pv.unique())
        self.modes = sorted(self.meta.beam_mode.unique())
        self.pv_numpy = pd.Categorical(self.meta.pv, categories=self.pvs).codes.astype(np.int64)
        self.mode_numpy = pd.Categorical(self.meta.beam_mode, categories=self.modes).codes.astype(np.int64)
        self.pv = torch.tensor(self.pv_numpy, device=self.device)
        self.mode = torch.tensor(self.mode_numpy, device=self.device)
        self.lengths = lengths
        self.splits = {s: np.flatnonzero(self.meta.split.eq(s)) for s in ["train", "validation", "calibration", "test"]}
        assert self.meta.groupby("cycle_id").split.nunique().max() == 1
        assert self.meta.groupby("source_archive").split.nunique().max() == 1
        assert set(self.meta.iloc[self.splits["train"]].pv) == set(self.pvs)
        assert set(self.meta.iloc[self.splits["train"]].beam_mode) == set(self.modes)
        identity = self.meta[["original_row", "pv", "cycle_id", "beam_mode", "length", "split"]].to_csv(index=False)
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        previous = self.output/"configuration.json"
        if previous.exists():
            config = json.loads(previous.read_text())
            if config["split_sha256"] != self.identity or config["seed"] != args.seed:
                raise ValueError("Dataset/splits changed; use another output directory")
        self.meta.to_csv(self.output/"sample_splits.csv", index=False)
        self.raw = {}
        positions = np.empty(len(frame), dtype=np.int64)
        self.fixed256 = torch.from_numpy(np.stack([resize256(w) for w in arrays])[:, None]).to(self.device)
        for length in sorted(set(lengths)):
            idx = np.flatnonzero(lengths == length)
            values = np.stack(arrays[idx]).astype(np.float32, copy=False)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite waveform")
            self.raw[length] = torch.from_numpy(values[:, None]).to(self.device)
            positions[idx] = np.arange(len(idx))
        self.positions = torch.tensor(positions, device=self.device)
        del arrays, frame, values
        self.norms = {}
        for grid in ["native", "256"]:
            means, stds = [], []
            for code in range(len(self.pvs)):
                indices = self.splits["train"][self.pv_numpy[self.splits["train"]] == code]
                total, squares, count = 0., 0., 0
                for batch in self.batches(indices, grid):
                    x = self.physical_batch(batch, grid).double()
                    total += x.sum().item()
                    squares += x.square().sum().item()
                    count += x.numel()
                mean = total/count
                std = max(squares/count-mean**2, 0.)**.5
                means.append(mean)
                stds.append(std if std >= 1e-6 else 1.)
            self.norms[grid] = (torch.tensor(means, dtype=torch.float32, device=self.device),
                                torch.tensor(stds, dtype=torch.float32, device=self.device))
            pd.DataFrame({"pv": self.pvs, "mean": means, "std": stds}).to_csv(self.output/f"normalization_{grid}.csv", index=False)
        train256 = self.fixed256[self.tensor(self.splits["train"])]
        self.global_mean = train256.mean().item()
        self.global_std = train256.std(correction=0).item()
        q = np.stack(self.sample.Q).astype(np.float32)
        train_idx = np.asarray(split["train_idx"])
        self.official_mean = float(q[train_idx].mean())
        self.official_std = float(q[train_idx].std()+1e-8)
        self.official_q = q
        self.config = dict(**vars(args), pvs=self.pvs, modes=self.modes, split_sha256=self.identity,
            counts={s: len(v) for s, v in self.splits.items()},
            cycles={s: int(self.meta.iloc[v].cycle_id.nunique()) for s, v in self.splits.items()},
            global_256_mean=self.global_mean, global_256_std=self.global_std,
            official_mean=self.official_mean, official_std=self.official_std,
            official_holdout_cycle_overlap=0, official_holdout_cycle_ids=sorted(self.official_cycles),
            interpolation="Full-span linear interpolation including both endpoints; NumPy float64 then float32",
            training_precision={"native_convolution": "bfloat16 autocast", "conv256_and_pca_ae": "float32", "evaluation": "float32"},
            pca_missing_tail="Training-only observed-position mean imputation; reconstruction errors use observed samples only",
            threshold="Median + 5 * population std (ddof=0) of independent calibration waveform errors",
            native_score="Full native-grid reconstruction MSE divided by fixed training-only per-PV native variance",
            source_note="Upstream normal-candidate selection used full-data PCA+kNN; cycle exclusion does not undo that selection dependence",
            official_note="Only 256-grid models receive official MSE; metadata-conditioned scores are identified explicitly",
            wandb_upload_performed=False)
        write_json(self.output/"configuration.json", self.config)
        for name, idx in self.splits.items():
            self.log(f"{name}: {len(idx):,} waveforms / {self.meta.iloc[idx].cycle_id.nunique()} cycles")

    def log(self, message):
        line = f"[{time.monotonic()-self.started:.1f}s] {message}"
        print(line, flush=True)
        with (self.output/"run.log").open("a") as stream:
            stream.write(line+"\n")

    def tensor(self, idx):
        return torch.as_tensor(idx, dtype=torch.long, device=self.device)

    def batches(self, indices, grid="native", shuffle=False, size=None):
        batches = []
        for length in (sorted(set(self.lengths[indices])) if grid == "native" else [None]):
            group = indices[self.lengths[indices] == length].copy() if length is not None else np.array(indices, copy=True)
            if shuffle:
                self.rng.shuffle(group)
            batch_size = size or self.args.batch_size
            batches.extend(group[start:start+batch_size] for start in range(0, len(group), batch_size))
        if shuffle:
            self.rng.shuffle(batches)
        return batches

    def physical_batch(self, idx, grid):
        ids = self.tensor(idx)
        if grid == "256":
            return self.fixed256[ids]
        length = int(self.lengths[idx[0]])
        assert np.all(self.lengths[idx] == length)
        return self.raw[length][self.positions[ids]]

    def normalized_batch(self, idx, grid):
        means, stds = self.norms[grid]
        pv = self.pv[self.tensor(idx)]
        return (self.physical_batch(idx, grid)-means[pv, None, None])/stds[pv, None, None]

    def model_for(self, name):
        if name in ["conv256", "conv256_context"]:
            return SmallConvAE(len(self.pvs), len(self.modes), conditioned=name.endswith("context"))
        if name.startswith("pca"):
            return PrincipalComponentAE(len(self.pvs), len(self.modes), int(name.split("_")[-2]))
        return WaveformAE(len(self.pvs), len(self.modes))

    def pca_key(self, name):
        return "256" if name.startswith("pca256_") else "native"

    def dimension(self, name):
        return int(name.split("_")[1])

    def target(self, idx, name):
        if name == "conv256":
            return (self.physical_batch(idx, "256")-self.global_mean)/self.global_std
        if name == "conv256_context":
            return self.normalized_batch(idx, "256")
        if name.startswith("pca"):
            pca = self.pcas[self.pca_key(name)]
            k = self.dimension(name)
            return (pca["pcs"][self.tensor(idx), :k]-pca["pc_mean"][:k])/pca["pc_std"][:k]
        return represent(self.normalized_batch(idx, "native"), name)

    def reconstruct(self, idx, name, prediction, native=True):
        ids = self.tensor(idx)
        pv = self.pv[ids]
        grid = "native"
        if name == "conv256":
            values = prediction*self.global_std+self.global_mean
            grid = "256"
        else:
            if name.startswith("pca"):
                grid = self.pca_key(name)
                pca = self.pcas[grid]
                k = self.dimension(name)
                pcs = prediction*pca["pc_std"][:k]+pca["pc_mean"][:k] if name.endswith("_ae") else prediction
                length = 256 if grid == "256" else int(self.lengths[idx[0]])
                prediction = (pcs@pca["basis"][:length, :k].T+pca["mean"][:length])[:, None]
            elif name == "conv256_context":
                grid = "256"
            else:
                prediction = native_reconstruction(prediction, name, int(self.lengths[idx[0]]))
            means, stds = self.norms[grid]
            values = prediction*stds[pv, None, None]+means[pv, None, None]
        if grid == "256" and native:
            values = F.interpolate(values, size=int(self.lengths[idx[0]]), mode="linear", align_corners=True)
        return values

    def native_error(self, idx, reconstructed):
        std = self.norms["native"][1][self.pv[self.tensor(idx)], None, None]
        return ((self.physical_batch(idx, "native")-reconstructed)/std).square().mean((1, 2))

    def curves(self, name, history):
        history = pd.DataFrame(history)
        history.to_csv(self.output/f"{name}_history.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for key, label in [("train_mse", "Training"), ("validation_mse", "Validation")]:
            axes[0].plot(history.epoch, history[key], label=label)
        best = history.loc[history.validation_mse.idxmin(), "epoch"]
        axes[0].axvline(best, color="gray", ls="--", lw=1, label=f"Selected epoch {best}")
        axes[0].set(title=name, xlabel="Epoch", ylabel="Training-target MSE", yscale="log")
        axes[0].legend()
        axes[1].plot(history.epoch, history.validation_waveform_mse, color="tab:orange")
        axes[1].set(title="Validation on the full native waveform", xlabel="Epoch", ylabel="Fixed per-PV normalized MSE", yscale="log")
        fig.savefig(self.output/"learning_curves"/f"{name}.png", dpi=160)
        plt.close(fig)

    def train(self, name):
        torch.manual_seed(self.args.seed)
        self.rng = np.random.default_rng(self.args.seed)
        model = self.model_for(name).to(self.device)
        completed = self.output/f"{name}.pt"
        checkpoint_path = self.output/f"{name}_checkpoint.pt"
        if completed.exists():
            saved = torch.load(completed, map_location=self.device, weights_only=False)
            extend = False
            if self.args.extend_unfinished and checkpoint_path.exists() and len(saved["history"]) < self.args.epochs:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                extend = checkpoint["stale"] < self.args.patience
                del checkpoint
            if not extend:
                model.load_state_dict(saved["state_dict"])
                self.log(f"Reusing completed {name}")
                return model.eval(), saved["history"], saved["training_seconds"]
            self.log(f"Extending {name} from epoch {len(saved['history'])}; validation had not stopped improving")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3, min_lr=1e-5)
        best, state, stale, history, first = float("inf"), None, 0, [], 1
        elapsed = 0.
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(saved["current_state"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            best, state, stale, history = saved["best"], saved["state_dict"], saved["stale"], saved["history"]
            first = saved["epoch"]+1
            elapsed = saved["training_seconds"]
            self.rng.bit_generator.state = saved["rng_state"]
        started = time.monotonic()-elapsed
        native_model = name in ["raw", "decimate4", "mean4", "mean16"]
        self.log(f"Starting {name}: {sum(p.numel() for p in model.parameters()):,} parameters")
        for epoch in range(first, self.args.epochs+1):
            epoch_started = time.monotonic()
            record = {"epoch": epoch}
            for split in ["train", "validation"]:
                model.train(split == "train")
                total, native_total, count = 0., 0., 0
                batches = self.batches(self.splits[split], shuffle=split == "train")
                with torch.set_grad_enabled(split == "train"):
                    for idx in batches:
                        ids = self.tensor(idx)
                        target = self.target(idx, name)
                        with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                            enabled=self.device.type == "cuda" and native_model):
                            prediction = model(target, self.pv[ids], self.mode[ids])
                        loss = (prediction.float()-target).square().mean()
                        if not torch.isfinite(loss):
                            raise ValueError(f"Nonfinite loss: {name}, epoch {epoch}")
                        if split == "train":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                        else:
                            native_total += self.native_error(idx, self.reconstruct(idx, name, prediction.float())).sum().item()
                        total += loss.detach().item()*len(idx)
                        count += len(idx)
                record[f"{split}_mse"] = total/count
                if split == "validation":
                    record["validation_waveform_mse"] = native_total/count
            record["learning_rate"] = optimizer.param_groups[0]["lr"]
            record["epoch_seconds"] = time.monotonic()-epoch_started
            history.append(record)
            scheduler.step(record["validation_mse"])
            significant = record["validation_mse"] < best*(1-1e-4)
            if record["validation_mse"] < best:
                best, state = record["validation_mse"], copy.deepcopy(model.state_dict())
            stale = 0 if significant else stale+1
            snapshot = dict(state_dict=state, current_state=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), best=best, stale=stale, history=history, epoch=epoch,
                training_seconds=time.monotonic()-started, rng_state=self.rng.bit_generator.state,
                split_sha256=self.identity, method=name)
            torch.save(snapshot, checkpoint_path)
            pd.DataFrame(history).to_csv(self.output/f"{name}_history.csv", index=False)
            if epoch == 1 or epoch % 5 == 0 or stale >= self.args.patience or epoch == self.args.epochs:
                self.curves(name, history)
            self.log(f"{name} epoch {epoch}: train={record['train_mse']:.6g}, validation={record['validation_mse']:.6g}, native_validation={record['validation_waveform_mse']:.6g}")
            write_json(self.output/"progress.json", dict(method=name, epoch=epoch,
                max_epochs=self.args.epochs, completed_methods=len(self.results), phase="training"))
            if stale >= self.args.patience:
                break
        model.load_state_dict(state)
        duration = time.monotonic()-started
        self.curves(name, history)
        torch.save(dict(state_dict=state, history=history, training_seconds=duration,
                        split_sha256=self.identity, method=name), completed)
        return model.eval(), history, duration

    @torch.no_grad()
    def fit_pca(self, grid):
        if grid in self.pcas:
            return
        self.log(f"Fitting training-only {grid}-grid PCA")
        started = time.monotonic()
        path = self.output/f"pca_{grid}.npz"
        if path.exists():
            with np.load(path) as saved:
                pca = {key: torch.tensor(saved[key], device=self.device) for key in ["mean", "basis", "pc_mean", "pc_std"]}
                pca["explained_variance"] = saved["explained_variance"]
        else:
            train = self.splits["train"]
            maximum = 256 if grid == "256" else int(self.lengths.max())
            matrix = torch.zeros((len(train), maximum), device=self.device)
            count = torch.zeros(maximum, device=self.device)
            position = 0
            for idx in self.batches(train, grid, size=256):
                x = self.normalized_batch(idx, grid)[:, 0]
                matrix[position:position+len(idx), :x.shape[1]] = x
                count[:x.shape[1]] += len(idx)
                position += len(idx)
            mean = matrix.sum(0)/count.clamp(min=1)
            position = 0
            for idx in self.batches(train, grid, size=256):
                length = 256 if grid == "256" else int(self.lengths[idx[0]])
                matrix[position:position+len(idx), :length] -= mean[:length]
                position += len(idx)
            energy = matrix.square().sum().item()
            torch.manual_seed(self.args.seed)
            u, singular, basis = torch.pca_lowrank(matrix, q=144, center=False, niter=3)
            pca = {"mean": mean, "basis": basis[:, :128].contiguous(),
                   "explained_variance": (singular[:128].square()/energy).cpu().numpy()}
            del matrix, u, singular, basis
        pcs = torch.zeros((len(self.meta), 128), device=self.device)
        for idx in self.batches(np.arange(len(self.meta)), grid, size=256):
            x = self.normalized_batch(idx, grid)[:, 0]
            pcs[self.tensor(idx)] = (x-pca["mean"][:x.shape[1]])@pca["basis"][:x.shape[1]]
        pca["pcs"] = pcs
        if "pc_mean" not in pca:
            training = pcs[self.tensor(self.splits["train"])]
            pca["pc_mean"] = training.mean(0)
            pca["pc_std"] = training.std(0, correction=0).clamp(min=1e-6)
            np.savez(path, **{key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in pca.items() if key != "pcs"})
        self.pcas[grid] = pca
        self.log(f"{grid}-grid PCA ready in {time.monotonic()-started:.1f}s; retained variance={pca['explained_variance'].sum():.3%}")

    @torch.no_grad()
    def evaluate(self, name, model=None, history=None, duration=0.):
        scores = np.zeros(len(self.meta), dtype=np.float64)
        previews = {}
        sample_positions = []
        for length in sorted(set(self.lengths)):
            candidates = self.splits["test"][self.lengths[self.splits["test"]] == length]
            sample_positions.extend(candidates[:2].tolist())
        for idx in self.batches(np.arange(len(self.meta))):
            ids = self.tensor(idx)
            if model is None:
                prediction = self.pcas[self.pca_key(name)]["pcs"][ids, :self.dimension(name)]
            else:
                prediction = model(self.target(idx, name), self.pv[ids], self.mode[ids]).float()
            reconstructed = self.reconstruct(idx, name, prediction)
            scores[idx] = self.native_error(idx, reconstructed).cpu().numpy()
            for k, position in enumerate(idx):
                if position in sample_positions:
                    previews[f"original_{position}"] = self.physical_batch(idx, "native")[k, 0].cpu().numpy()
                    previews[f"reconstructed_{position}"] = reconstructed[k, 0].cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite scores")
        calibration = scores[self.splits["calibration"]]
        median, sigma = float(np.median(calibration)), float(np.std(calibration, ddof=0))
        cutoff = median+5*sigma
        np.save(self.output/f"{name}_scores.npy", scores)
        np.savez(self.output/f"{name}_reconstructions.npz", **previews)
        waves, cycles = CycleDiagnoser(cutoff).predict(self.meta.assign(score=scores))
        cycles.to_csv(self.output/f"{name}_cycle_predictions.csv", index=False)
        waves.loc[waves.pv_anomaly_candidate].to_csv(self.output/f"{name}_candidate_pvs.csv", index=False)
        test = self.splits["test"]
        test_cycles = cycles.loc[cycles["split"].eq("test")]
        row = dict(method=name, epochs=len(history) if history else 0,
            best_epoch=int(min(history, key=lambda r: r["validation_mse"])["epoch"]) if history else None,
            epoch_limit_reached=bool(history and len(history) >= self.args.epochs),
            parameters=sum(p.numel() for p in model.parameters()) if model is not None else 0,
            training_seconds=float(duration), test_waveforms=len(test),
            test_native_mse=float(scores[test].mean()), test_candidate_waveforms=int((scores[test] > cutoff).sum()),
            test_candidate_rate=float((scores[test] > cutoff).mean()), test_cycles=len(test_cycles),
            test_candidate_cycles=int(test_cycles.anomaly_candidate.eq(True).sum()),
            calibration_median=median, calibration_std=sigma, threshold=cutoff,
            input_conditions="waveform only; global training normalization" if name == "conv256" else
                ("waveform only; per-PV training normalization" if model is None else "waveform + PV + beam_mode"))
        self.results.append(row)
        write_json(self.output/f"{name}_metrics.json", row)
        self.plot_reconstructions(name, previews)
        self.publish()
        self.log(f"Finished {name}: test native MSE={row['test_native_mse']:.6g}, candidates={row['test_candidate_waveforms']}/{len(test)}")

    def plot_reconstructions(self, name, previews):
        keys = sorted(int(k.split("_")[1]) for k in previews if k.startswith("original_"))
        fig, axes = plt.subplots(len(keys), 1, figsize=(12, 3*len(keys)), squeeze=False, layout="constrained")
        for ax, position in zip(axes[:, 0], keys):
            original, recon = previews[f"original_{position}"], previews[f"reconstructed_{position}"]
            step = max(1, len(original)//5000)
            index = np.arange(0, len(original), step)
            ax.plot(index, original[::step], lw=.8, label="Original")
            ax.plot(index, recon[::step], lw=.8, alpha=.8, label="Reconstruction")
            row = self.meta.iloc[position]
            ax.set(title=f"{name} | {row.pv} | cycle {row.cycle_id} | {row.beam_mode}", xlabel="Native sample index", ylabel="Recorded value")
            ax.legend()
        fig.savefig(self.output/f"{name}_reconstructions.png", dpi=150)
        plt.close(fig)

    @torch.no_grad()
    def official_evaluate(self, name, model=None):
        held = self.sample.iloc[self.official_idx]
        pvs = held.pv_base.str.replace("CmpSDS", "Cmp1SDS", regex=False).str.replace(r"-(?:PMData|DODData)$", "", case=False, regex=True)
        pv = torch.tensor(pd.Categorical(pvs, categories=self.pvs).codes.astype(np.int64), device=self.device)
        mode = torch.tensor(pd.Categorical(held.beam_mode, categories=self.modes).codes.astype(np.int64), device=self.device)
        if (pv < 0).any() or (mode < 0).any():
            raise ValueError("Official input has unseen metadata categories")
        x = torch.tensor(self.official_q[self.official_idx, None], device=self.device)
        if name == "conv256":
            reconstructed = model((x-self.global_mean)/self.global_std, pv, mode)*self.global_std+self.global_mean
        else:
            mean, std = self.norms["256"]
            normalized = (x-mean[pv, None, None])/std[pv, None, None]
            if name == "conv256_context":
                prediction = model(normalized, pv, mode)
            else:
                pca = self.pcas["256"]
                k = self.dimension(name)
                pcs = (normalized[:, 0]-pca["mean"])@pca["basis"][:, :k]
                if model is not None:
                    pcs = model((pcs-pca["pc_mean"][:k])/pca["pc_std"][:k], pv, mode)*pca["pc_std"][:k]+pca["pc_mean"][:k]
                prediction = (pcs@pca["basis"][:, :k].T+pca["mean"])[:, None]
            reconstructed = prediction*std[pv, None, None]+mean[pv, None, None]
        # Identical normalization and per-event averaging to score.py.
        reference_x = (x-self.official_mean)/self.official_std
        reference_recon = (reconstructed-self.official_mean)/self.official_std
        errors = (reference_x-reference_recon).square().mean((1, 2)).cpu().numpy()
        detail = held[["cycle_id", "pv_base", "beam_mode"]].copy()
        detail.insert(0, "sample_row", self.official_idx)
        detail["mse"] = errors
        detail.to_csv(self.output/f"{name}_official_holdout_scores.csv", index=False)
        row = dict(method=name, holdout_events=len(errors), official_holdout_mse=float(errors.mean()),
            reference_bar=.0194, below_reference=bool(errors.mean() < .0194),
            metadata_inputs=name != "conv256", upstream_normal_selection_independent=False)
        self.official_results.append(row)
        np.savez(self.output/f"{name}_official_reconstructions.npz", Q=x[:, 0].cpu().numpy(),
                 reconstruction=reconstructed[:, 0].cpu().numpy(), sample_rows=self.official_idx)
        self.publish()
        self.log(f"Official 42-event MSE ({name}): {row['official_holdout_mse']:.6g}")

    def publish(self):
        table = pd.DataFrame(self.results)
        table.to_csv(self.output/"comparison.csv", index=False)
        pd.DataFrame(self.official_results).to_csv(self.output/"official_holdout_comparison.csv", index=False)
        write_json(self.output/"summary.json", dict(results=self.results, official_results=self.official_results,
            note="Candidate rates are not validated false-positive rates. No confirmed abnormal MBL waveforms were supplied. Upstream full-data PCA+kNN selection limits independent official-score interpretation."))
        if len(table):
            fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
            for ax, column, label in zip(axes, ["test_native_mse", "test_candidate_rate", "training_seconds"],
                                        ["Native-grid normalized MSE", "Unlabelled test candidate fraction", "AE training seconds (PCA fit excluded)"]):
                ax.bar(table.method, table[column])
                ax.set_ylabel(label)
                ax.tick_params(axis="x", rotation=80)
            axes[0].set_yscale("log")
            fig.savefig(self.output/"comparison.png", dpi=160)
            plt.close(fig)

    def run(self):
        methods = ["conv256", "conv256_context", "raw", "decimate4", "mean4", "mean16"]
        for name in methods:
            model, history, duration = self.train(name)
            self.evaluate(name, model, history, duration)
            if name.startswith("conv256"):
                self.official_evaluate(name, model)
            del model
        for grid in ["native", "256"]:
            self.fit_pca(grid)
            for dimension in [32, 64, 128]:
                prefix = "pca" if grid == "native" else "pca256"
                baseline = f"{prefix}_{dimension}"
                self.evaluate(baseline)
                if grid == "256":
                    self.official_evaluate(baseline)
                name = baseline+"_ae"
                model, history, duration = self.train(name)
                self.evaluate(name, model, history, duration)
                if grid == "256":
                    self.official_evaluate(name, model)
                del model
        write_json(self.output/"progress.json", dict(phase="complete", completed_methods=len(self.results),
            elapsed_seconds=time.monotonic()-self.started))
        self.log("All MBL AE comparisons complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(HERE/"mbl_dwn0_cmp1_normal_without_official_holdout.pkl"))
    parser.add_argument("--output", default=str(HERE/"mbl_autoencoder_comparison"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--extend-unfinished", action="store_true", help="Resume completed runs that reached the previous epoch budget without early stopping")
    args = parser.parse_args()
    comparison = Comparison(args)
    if not args.prepare_only:
        comparison.run()

"""Moments at the planes: settings + plane_mm -> the 13 moments, physical units, DataFrames in and out.

MLPEnsemble: MLPs on the 13 targets on the scorer's scale (use_physics adds four physics inputs).
FiLMEnsemble: the plane through a network the settings scale and shift; it gives a 6D Gaussian (means, full
covariance) and the charge, trained with the Gaussian NLL (+ lambda_c x MSE of the transverse centroids).
Hybrid: one model's prediction with some columns averaged from others.
fit(train, val) keeps each net's best validation epoch; refit(rows) retrains on all rows for those epochs.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

import pallas_score

SETTINGS = ["P_max", "cN2_max", "L_inj", "dip_frac", "x_of"]
COORDS = ["zeta", "y", "z", "ux", "uy", "uz"]
targets = [f"mean_{c}" for c in COORDS] + [f"sigma_{c}" for c in COORDS] + ["q_tot_pC"]
centroids = ["mean_y", "mean_z", "mean_uy", "mean_uz"]
physics_names = ["n_plateau", "z_from_end", "n_e", "dE_gain"]

PARAMS = {"mlp": {"hidden": 256, "n_layers": 4, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 1024,
                  "epochs": 400, "n_nets": 5, "clip": None, "use_physics": False, "seed": 0, "device": "cpu"},
          "film": {"hidden": 128, "n_layers": 4, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 1024,
                   "epochs": 400, "n_nets": 5, "clip": 5.0, "lambda_c": 0.0, "weighted": True, "seed": 0,
                   "device": "cpu"}}


def build_model(kind="mlp", **params):
    cls = {"mlp": MLPEnsemble, "film": FiLMEnsemble}[kind]
    return cls(**{**PARAMS[kind], **params})


def physics_features(df):
    """n_plateau (per config), z_from_end, n_e, dE_gain at the plane (the plane as z)."""
    import physics_formulas as pf

    knobs = df[["P_max", "cN2_max", "L_inj", "dip_frac"]].to_numpy(np.float64)
    z = df.plane_mm.to_numpy(np.float64)
    keys, inverse = np.unique(knobs, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    x = np.linspace(0, 7.2, 3601)
    out = np.empty((len(df), 4))
    for k, (P, c, L, d) in enumerate(keys):
        end = L + pallas_score.PLASMA_END_OFFSET_MM
        n_He, n_N = pf.species_B(x * 1e3, P, c, L, d)
        n_e = pf.m3_to_cm3(pf.electrons_background(n_He, n_N))
        n_e[x > end] = 0.0
        gain = pf.energy_gain_MeV(x * 1e3, n_e, eta=1.0)
        rows = inverse == k
        out[rows, 0] = n_e.max() / 1e18
        out[rows, 1] = z[rows] - end
        out[rows, 2] = np.interp(z[rows], x, n_e) / 1e18
        out[rows, 3] = np.interp(z[rows], x, gain)
    return pd.DataFrame(out, columns=physics_names, index=df.index)


class Ensemble:
    def __init__(self, **params):
        self.params = params

    def _tensor(self, a, device=None):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device or self.params["device"])

    def _train(self, df, val, epochs, seed):
        """With val: keep the epoch of lowest validation loss. Without: exactly `epochs` epochs."""
        p = self.params
        torch.manual_seed(seed)
        net = self._net().to(p["device"])
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        data = self._tensors(df)
        data_val = self._tensors(val) if val is not None else None
        best, kept, state = float("inf"), epochs, None
        n = len(next(iter(data.values())))
        for epoch in range(epochs):
            net.train()
            for idx in torch.randperm(n, device=p["device"]).split(p["batch_size"]):
                loss = self._loss(net, {k: v[idx] for k, v in data.items()})
                opt.zero_grad()
                loss.backward()
                if p["clip"]:
                    nn.utils.clip_grad_norm_(net.parameters(), p["clip"])
                opt.step()
            sched.step()
            if data_val is None:
                continue
            net.eval()
            with torch.no_grad():
                v = self._loss(net, data_val).item()
            if v < best:
                best, kept, state = v, epoch + 1, {k: w.clone() for k, w in net.state_dict().items()}
        if state is not None:
            net.load_state_dict(state)
            print(f"  kept epoch {kept}, validation loss {best:.4f}", flush=True)
        return net.cpu().eval(), kept

    def fit(self, train, val):
        self._setup(train)
        p = self.params
        trained = [self._train(train, val, p["epochs"], p["seed"] + k) for k in range(p["n_nets"])]
        self.nets, self.epochs = [t[0] for t in trained], [t[1] for t in trained]
        return self

    def refit(self, rows):
        p = self.params
        self.nets = [self._train(rows, None, e, p["seed"] + k)[0] for k, e in enumerate(self.epochs)]
        return self


# ---- MLP ----------------------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, n_layers):
        super().__init__()
        body, width = [], n_in
        for _ in range(n_layers):
            body += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.net = nn.Sequential(*body, nn.Linear(width, n_out))

    def forward(self, x):
        return self.net(x)


def to_scale(df):
    return np.stack([pallas_score.physical_to_normalised(t, df[t]) for t in targets], 1)


class MLPEnsemble(Ensemble):
    def _features(self, df):
        x = df[SETTINGS + ["plane_mm"]]
        if self.params["use_physics"]:
            x = pd.concat([x, physics_features(df)], axis=1)
        return x.to_numpy(np.float64)

    def _setup(self, train):
        self.x_scaler = StandardScaler().fit(self._features(train))
        self.y_scaler = StandardScaler().fit(to_scale(train))

    def _net(self):
        p = self.params
        return MLP(len(SETTINGS) + 1 + 4 * p["use_physics"], len(targets), p["hidden"], p["n_layers"])

    def _tensors(self, df):
        return {"x": self._tensor(self.x_scaler.transform(self._features(df))),
                "y": self._tensor(self.y_scaler.transform(to_scale(df)))}

    def _loss(self, net, b):
        return nn.functional.mse_loss(net(b["x"]), b["y"])

    def predict(self, df):
        x = self._tensor(self.x_scaler.transform(self._features(df)), "cpu")
        with torch.no_grad():
            mean = torch.stack([net(x) for net in self.nets]).mean(0).numpy()
        a = self.y_scaler.inverse_transform(mean)
        return pd.DataFrame({t: pallas_score.normalised_to_physical(t, a[:, j]) for j, t in enumerate(targets)})


# ---- FiLM Gaussian ------------------------------------------------------------------------------

tril = torch.tril_indices(6, 6)
c_index = [COORDS.index(c.removeprefix("mean_")) for c in centroids]


def cov_of(df):
    C = np.zeros((len(df), 6, 6))
    for i in range(6):
        for j in range(i, 6):
            C[:, i, j] = C[:, j, i] = df[f"cov_{i}{j}"].to_numpy()
    return C


def means_of(df):
    return df[[f"mean_{c}" for c in COORDS]].to_numpy(np.float64)


class FiLMGauss(nn.Module):
    def __init__(self, hidden, n_layers):
        super().__init__()
        self.hidden, self.n_layers = hidden, n_layers
        self.p_in = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.cond = nn.Sequential(nn.Linear(len(SETTINGS), hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2 * hidden * n_layers))
        self.out = nn.Linear(hidden, 6 + 21 + 1)
        self.act = nn.SiLU()

    def forward(self, s, p):
        gb = self.cond(s).view(-1, self.n_layers, 2, self.hidden)
        h = self.act(self.p_in(p))
        for i, layer in enumerate(self.layers):
            h = self.act((1 + gb[:, i, 0]) * layer(h) + gb[:, i, 1])
        o = self.out(h)
        L = torch.zeros(len(o), 6, 6, device=o.device)
        L[:, tril[0], tril[1]] = o[:, 6:27]
        d = torch.arange(6)
        L[:, d, d] = nn.functional.softplus(L[:, d, d]) + 1e-4
        return o[:, :6], L, o[:, 27]


class FiLMEnsemble(Ensemble):
    def _setup(self, train):
        self.centre = np.median(means_of(train), 0)
        self.scale = np.median(np.sqrt(np.diagonal(cov_of(train), axis1=1, axis2=2)), 0)
        self.s_scaler = StandardScaler().fit(train[SETTINGS].to_numpy(np.float64))
        self.p_scaler = StandardScaler().fit(train[["plane_mm"]].to_numpy(np.float64))
        logq = np.log(train.q_tot_pC.to_numpy())
        self.q_mu, self.q_sd, self.q_ref = logq.mean(), logq.std(), train.q_tot_pC.mean()
        self.c_sd = ((means_of(train) - self.centre) / self.scale)[:, c_index].std(0, ddof=1)

    def _net(self):
        return FiLMGauss(self.params["hidden"], self.params["n_layers"])

    def _inputs(self, df, device=None):
        return (self._tensor(self.s_scaler.transform(df[SETTINGS].to_numpy(np.float64)), device),
                self._tensor(self.p_scaler.transform(df[["plane_mm"]].to_numpy(np.float64)), device))

    def _tensors(self, df):
        s, p = self._inputs(df)
        return {"s": s, "p": p, "m": self._tensor((means_of(df) - self.centre) / self.scale),
                "S": self._tensor(cov_of(df) / np.outer(self.scale, self.scale)),
                "q": self._tensor((np.log(df.q_tot_pC.to_numpy()) - self.q_mu) / self.q_sd),
                "w": self._tensor(df.q_tot_pC.to_numpy() / self.q_ref)}

    def _loss(self, net, b):
        mu, L, q = net(b["s"], b["p"])
        diff = (b["m"] - mu).unsqueeze(-1)
        logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
        Linv = torch.linalg.solve_triangular(L, torch.eye(6, device=L.device).expand_as(L), upper=False)
        P = Linv.transpose(1, 2) @ Linv
        nll = 0.5 * (logdet + (P * b["S"]).sum((1, 2)) + (diff.transpose(1, 2) @ P @ diff).squeeze(-1).squeeze(-1))
        w = b["w"] if self.params["weighted"] else torch.ones_like(b["w"])
        loss = (w * nll).sum() / w.sum() + ((q - b["q"]) ** 2).mean()
        if self.params["lambda_c"]:
            c_sd = torch.as_tensor(self.c_sd, dtype=torch.float32, device=mu.device)
            loss = loss + self.params["lambda_c"] * (((mu - b["m"])[:, c_index] / c_sd) ** 2).mean()
        return loss

    def predict_gaussian(self, df):
        """Means (rows, 6), covariances (rows, 6, 6), charge (rows,), physical units."""
        s, p = self._inputs(df, "cpu")
        with torch.no_grad():
            outs = [net(s, p) for net in self.nets]
        mus = torch.stack([o[0] for o in outs])
        mu = mus.mean(0)
        cov = torch.stack([o[1] @ o[1].transpose(1, 2) for o in outs]).mean(0) \
            + ((mus - mu).unsqueeze(-1) @ (mus - mu).unsqueeze(-2)).mean(0)
        q = torch.stack([o[2] for o in outs]).mean(0).numpy()
        return (mu.numpy() * self.scale + self.centre, cov.numpy() * np.outer(self.scale, self.scale),
                np.exp(q * self.q_sd + self.q_mu))

    def predict(self, df):
        mu, cov, q = self.predict_gaussian(df)
        sd = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
        pred = pd.DataFrame({f"mean_{c}": mu[:, i] for i, c in enumerate(COORDS)})
        for i, c in enumerate(COORDS):
            pred[f"sigma_{c}"] = sd[:, i]
        pred["q_tot_pC"] = q
        return pred[targets]


# ---- Hybrid -------------------------------------------------------------------------------------

class Hybrid:
    """base's prediction, `columns` replaced by the average of the `sources`' predictions."""

    def __init__(self, base, columns, sources):
        self.base, self.columns, self.sources = base, list(columns), list(sources)

    def predict(self, df):
        pred = self.base.predict(df)
        pred[self.columns] = np.mean([s.predict(df)[self.columns].to_numpy() for s in self.sources], 0)
        return pred

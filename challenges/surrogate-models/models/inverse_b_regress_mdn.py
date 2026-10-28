"""Inverse B: an average of MLPs for the answer, a re-centred mixture around it for the probabilities.

Stage 1: `n_members` MLPs, bunch -> the four settings (standardised), MSE; their average m is the answer.
Stage 2: a network gives `k` solutions (weight, offset, width per setting) whose weighted mean offset is held at
zero, so the mixture's mean is m and R2 stays stage 1's. It trains on stage 1's out-of-fold answers: on rows it
saw, stage 1 is too accurate and the widths would come out too narrow.
fit() picks the epochs on `val_fraction` of the rows; refit() retrains on all rows with those epochs.
`n_spectrum_pca` > 0 adds principal components of the energy spectrum (columns spec_000 ...): internal only.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn

PARAMS = {"hidden": 256, "n_layers": 4, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 256,
          "epochs": 1500, "n_members": 5, "k": 5, "mdn_hidden": 128, "mdn_layers": 3,
          "mdn_lr": 1e-3, "mdn_weight_decay": 1e-4, "mdn_epochs": 600, "min_sd": 0.02, "n_folds": 5,
          "val_fraction": 0.25, "seed": 0, "threads": 16, "n_spectrum_pca": 0,
          "columns": ["de_mad", "div_rms", "e_med_mev", "i_peak", "n_emit_x", "q_end", "sigma_y",
                      "sigma_z", "x_of"],
          "log_columns": ["q_end", "i_peak", "n_emit_x", "sigma_y"]}


def build_model(**params):
    return RegressMixture(**{**PARAMS, **params})


def body(n_in, hidden, n_layers):
    layers, width = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(width, hidden), nn.SiLU()]
        width = hidden
    return nn.Sequential(*layers)


class Regress(nn.Module):
    def __init__(self, n_in, n_out, hidden, n_layers):
        super().__init__()
        self.net = nn.Sequential(body(n_in, hidden, n_layers), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class Recentred(nn.Module):
    """k solutions around a given answer m, their weighted mean offset held at zero."""

    def __init__(self, n_in, n_out, k, hidden, n_layers, min_sd):
        super().__init__()
        self.body = body(n_in, hidden, n_layers)
        self.logits = nn.Linear(hidden, k)
        self.offset = nn.Linear(hidden, k * n_out)
        self.raw_sd = nn.Linear(hidden, k * n_out)
        self.k, self.n_out, self.min_sd = k, n_out, min_sd

    def forward(self, x, m):
        h = self.body(x)
        shape = (-1, self.k, self.n_out)
        log_pi = torch.log_softmax(self.logits(h), -1)
        offset = self.offset(h).view(shape)
        offset = offset - (log_pi.exp()[..., None] * offset).sum(1, keepdim=True)
        sd = nn.functional.softplus(self.raw_sd(h)).view(shape) + self.min_sd
        return log_pi, m[:, None] + offset, sd


def mse(net, x, y):
    return nn.functional.mse_loss(net(x), y)


def mixture_nll(net, x, m, y):
    log_pi, mu, sd = net(x, m)
    log_n = (-0.5 * ((y[:, None] - mu) / sd) ** 2 - torch.log(sd) - 0.5 * np.log(2 * np.pi)).sum(2)
    return -torch.logsumexp(log_pi + log_n, 1).mean()


class RegressMixture:
    def __init__(self, **params):
        self.params = params

    def _x(self, X, fit=False):
        columns = self.params["columns"]
        x = np.asarray(X, np.float64).copy()
        spec = [j for j, c in enumerate(columns) if c.startswith("spec_")]
        bunch = [j for j, c in enumerate(columns) if not c.startswith("spec_")]
        for c in self.params["log_columns"]:
            j = columns.index(c)
            x[:, j] = np.log10(x[:, j])
        if fit:
            self.x_scaler = StandardScaler().fit(x[:, bunch])
        out = self.x_scaler.transform(x[:, bunch])
        n_pca = self.params.get("n_spectrum_pca", 0)
        if n_pca:
            shape = x[:, spec] / x[:, spec].sum(1, keepdims=True)
            if fit:
                self.pca = PCA(n_pca).fit(shape)
                self.pc_scaler = StandardScaler().fit(self.pca.transform(shape))
            out = np.hstack([out, self.pc_scaler.transform(self.pca.transform(shape))])
        return torch.tensor(out, dtype=torch.float32)

    def _y(self, Y, fit=False):
        y = np.asarray(Y, np.float64)
        if fit:
            self.y_scaler = StandardScaler().fit(y)
        return torch.tensor(self.y_scaler.transform(y), dtype=torch.float32)

    def _train(self, net, loss_of, fit_data, val_data, epochs, lr, weight_decay, seed):
        """With val_data: keep the epoch of lowest validation loss and return how many epochs
        that took. Without: train exactly `epochs` epochs."""
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        best, best_epoch, state = float("inf"), epochs - 1, None
        for epoch in range(epochs):
            net.train()
            for idx in torch.randperm(len(fit_data[0])).split(self.params["batch_size"]):
                loss = loss_of(net, *[d[idx] for d in fit_data])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            if val_data is None:
                continue
            net.eval()
            with torch.no_grad():
                val_loss = loss_of(net, *val_data).item()
            if val_loss < best:
                best, best_epoch = val_loss, epoch
                state = {k: v.clone() for k, v in net.state_dict().items()}
        if state is not None:
            net.load_state_dict(state)
        net.eval()
        return best_epoch + 1

    def _regress(self, n_in, n_out):
        return Regress(n_in, n_out, self.params["hidden"], self.params["n_layers"])

    def _members(self, x, y, val, epochs_each):
        p = self.params
        self.members, self.member_epochs = [], []
        for s, epochs in enumerate(epochs_each):
            net = self._regress(x.shape[1], y.shape[1])
            kept = self._train(net, mse, (x, y), val, epochs, p["lr"], p["weight_decay"], p["seed"] + s)
            self.members.append(net)
            self.member_epochs.append(kept)

    def _out_of_fold(self, x, y):
        """Stage 1's answer for each row from a network that did not train on it."""
        p = self.params
        fold = np.random.default_rng(p["seed"]).permutation(len(x)) % p["n_folds"]
        m = torch.zeros_like(y)
        for f in range(p["n_folds"]):
            fit, out = torch.tensor(fold != f), torch.tensor(fold == f)
            net = self._regress(x.shape[1], y.shape[1])
            self._train(net, mse, (x[fit], y[fit]), None, self.fold_epochs, p["lr"], p["weight_decay"], p["seed"] + f)
            with torch.no_grad():
                m[out] = net(x[out])
        return m

    def _mixture(self, x, m, y, val, epochs):
        p = self.params
        self.mixture = Recentred(x.shape[1], y.shape[1], p["k"], p["mdn_hidden"], p["mdn_layers"], p["min_sd"])
        self.mixture_epochs = self._train(self.mixture, mixture_nll, (x, m, y), val, epochs,
                                          p["mdn_lr"], p["mdn_weight_decay"], p["seed"])

    def fit(self, X, Y):
        p = self.params
        torch.set_num_threads(p["threads"])
        order = np.random.default_rng(p["seed"]).permutation(len(X))
        n_val = int(round(p["val_fraction"] * len(X)))
        val_rows, fit_rows = np.sort(order[:n_val]), np.sort(order[n_val:])
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        x, y = self._x(X[fit_rows], fit=True), self._y(Y[fit_rows], fit=True)
        x_val, y_val = self._x(X[val_rows]), self._y(Y[val_rows])

        self._members(x, y, (x_val, y_val), [p["epochs"]] * p["n_members"])
        self.fold_epochs = int(np.median(self.member_epochs))
        m_oof = self._out_of_fold(x, y)
        with torch.no_grad():
            m_val = self._answer(x_val)
        self._mixture(x, m_oof, y, (x_val, m_val, y_val), p["mdn_epochs"])
        self.val_error = np.sqrt(((m_val - y_val) ** 2).mean(0).numpy()) * self.y_scaler.scale_
        return self

    def refit(self, X, Y):
        """All rows, the epochs fit() kept, no validation."""
        torch.set_num_threads(self.params["threads"])
        x, y = self._x(X, fit=True), self._y(Y, fit=True)
        self._members(x, y, None, list(self.member_epochs))
        m_oof = self._out_of_fold(x, y)
        self._mixture(x, m_oof, y, None, self.mixture_epochs)
        return self

    def _answer(self, x):
        return torch.stack([net(x) for net in self.members]).mean(0)

    def predict(self, X):
        with torch.no_grad():
            m = self._answer(self._x(X)).numpy()
        return self.y_scaler.inverse_transform(m)

    def predict_mixture(self, X):
        """Weights (rows x k), centres and widths (rows x k x targets), physical units."""
        x = self._x(X)
        with torch.no_grad():
            log_pi, mu, sd = self.mixture(x, self._answer(x))
        scale, mean = self.y_scaler.scale_, self.y_scaler.mean_
        return log_pi.exp().numpy(), mu.numpy() * scale + mean, sd.numpy() * scale

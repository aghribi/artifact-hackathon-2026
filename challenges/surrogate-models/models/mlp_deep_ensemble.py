"""Trajectory: a deep ensemble of MLPs with their own uncertainty.

`n_members` networks from different random starts, each giving a value and a variance per target; the
ensemble mean is the prediction, the spread its uncertainty. Each keeps its best epoch on `val_fraction`
of the training configurations. `use_physics` adds the features in physics_names (or those in `physics`) from the
raw inputs; `drop` hides input columns from the network; `two_stage` trains the means with MSE first, then
only the variances.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

PARAMS = {"hidden": 64, "n_layers": 3, "n_members": 5, "lr": 2e-3, "weight_decay": 0.0, "epochs": 200, "warmup": 20,
          "batch_size": 2048, "val_fraction": 0.2, "use_physics": False, "physics": None, "drop": None,
          "two_stage": False, "spread_epochs": 100, "spread_lr": 3e-3,
          "columns": ["cn2_max", "dip_frac", "l_inj", "p_max", "x_of", "z_mm"], "device": None, "seed": 0}
physics_names = ["n_plateau", "z_from_end", "n_e", "dE_gain"]

class Net(nn.Module):
    """A plain MLP on the settings and z together."""

    def __init__(self, n_config, n_position, n_out, hidden, n_layers):
        super().__init__()
        layers, width = [], n_config + n_position
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.net = nn.Sequential(*layers, nn.Linear(width, 2 * n_out))
        self.n_out = n_out

    def forward(self, s, p):
        out = self.net(torch.cat([s, p], dim=1))
        return out[:, :self.n_out], nn.functional.softplus(out[:, self.n_out:]) + 1e-6


def build_model(**params):
    return DeepEnsemble(**{**PARAMS, **params})


def physics_features(raw, columns):
    """n_plateau (per config), z_from_end, n_e, dE_gain (per position), from the raw inputs."""
    import pallas_score
    import physics_formulas as pf

    col = {c: i for i, c in enumerate(columns)}
    keys, inverse = np.unique(raw[:, [col["p_max"], col["cn2_max"], col["l_inj"], col["dip_frac"]]], axis=0,
                              return_inverse=True)
    inverse = inverse.ravel()
    z = raw[:, col["z_mm"]]
    x = np.linspace(0, 7.2, 3601)
    out = np.empty((len(raw), 4))
    for k, (P, c, L, d) in enumerate(keys):
        end = L + pallas_score.PLASMA_END_OFFSET_MM
        n_e = pf.m3_to_cm3(pf.electrons_background(*pf.species_B(x * 1e3, P, c, L, d)))
        n_e[x > end] = 0.0
        rows = inverse == k
        out[rows, 0] = n_e.max() / 1e18
        out[rows, 1] = z[rows] - end
        out[rows, 2] = np.interp(z[rows], x, n_e) / 1e18
        out[rows, 3] = np.interp(z[rows], x, pf.energy_gain_MeV(x * 1e3, n_e, eta=1.0))
    return out


class DeepEnsemble:
    def __init__(self, hidden, n_layers, n_members, lr, weight_decay, epochs, warmup, batch_size, val_fraction,
                 use_physics, physics, drop, two_stage, spread_epochs, spread_lr, columns, device, seed):
        self.hidden, self.n_layers, self.n_members = hidden, n_layers, n_members
        self.lr, self.weight_decay, self.epochs = lr, weight_decay, epochs
        self.warmup, self.batch_size, self.val_fraction = warmup, batch_size, val_fraction
        self.use_physics, self.physics = use_physics, list(physics) if physics else None
        self.columns = [c.lower() for c in columns]
        self.drop = [c.lower() for c in drop] if drop else None
        self.two_stage, self.spread_epochs, self.spread_lr = two_stage, spread_epochs, spread_lr
        self.device, self.seed, self.spread_scale = device, seed, None

    def _split_inputs(self, X):
        """(per-config inputs, per-position inputs): z_mm and the per-position physics are the latter."""
        X = np.asarray(X, np.float64)
        z_col = self.columns.index("z_mm")
        s = np.delete(X, [z_col] + [self.columns.index(c) for c in getattr(self, "drop", None) or []], axis=1)
        p = X[:, [z_col]]
        if self.use_physics:
            phys = physics_features(X, self.columns)
            keep = getattr(self, "physics", None) or physics_names
            s = np.hstack([s, phys[:, [physics_names.index(f) for f in keep if f == "n_plateau"]]])
            p = np.hstack([p, phys[:, [physics_names.index(f) for f in keep if f != "n_plateau"]]])
        return s, p

    def _on(self):
        on = self.device or "cuda"
        if on.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return on

    def fit(self, X, Y):
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        s, p = self._split_inputs(X)
        groups = np.unique(np.delete(X, self.columns.index("z_mm"), axis=1), axis=0, return_inverse=True)[1].ravel()
        val_groups = np.random.default_rng(self.seed).choice(groups.max() + 1, size=max(1, round(self.val_fraction * (groups.max() + 1))),
                                                             replace=False)
        is_val = np.isin(groups, val_groups)
        self.s_scaler = StandardScaler().fit(s[~is_val])
        self.p_scaler = StandardScaler().fit(p[~is_val])
        self.y_scaler = StandardScaler().fit(Y[~is_val])
        tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=self._on())
        data = {"s_train": tensor(self.s_scaler.transform(s[~is_val])), "p_train": tensor(self.p_scaler.transform(p[~is_val])),
                "y_train": tensor(self.y_scaler.transform(Y[~is_val])), "s_val": tensor(self.s_scaler.transform(s[is_val])),
                "p_val": tensor(self.p_scaler.transform(p[is_val])), "y_val": tensor(self.y_scaler.transform(Y[is_val]))}
        print(f"{groups.max() + 1 - len(val_groups)} train / {len(val_groups)} validation configurations, "
              f"{s.shape[1]} + {p.shape[1]} inputs, on {self._on()}")
        self.members, self.histories, self.best_epochs = [], [], []
        for k in range(self.n_members):
            net, history, best_epoch = self._train_one(self.seed + k, data)
            if self.two_stage:
                self._fit_spread(net, data)
            print(f"member {k}: epoch {best_epoch}")
            self.members.append(net.to("cpu"))
            self.histories.append(history)
            self.best_epochs.append(best_epoch)
        return self

    def _train_one(self, seed, data):
        """MSE for `warmup` epochs (all of them with two_stage), then Gaussian NLL; best validation epoch."""
        on = self._on()
        torch.manual_seed(seed)
        net = Net(data["s_train"].shape[1], data["p_train"].shape[1], data["y_train"].shape[1], self.hidden, self.n_layers).to(on)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        nll, mse = nn.GaussianNLLLoss(), nn.MSELoss()
        history, best, best_epoch, best_state = [], float("inf"), 0, None
        n = len(data["y_train"])
        for epoch in range(self.epochs):
            net.train()
            order, total = torch.randperm(n, device=on), 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                mean, var = net(data["s_train"][idx], data["p_train"][idx])
                y = data["y_train"][idx]
                loss = mse(mean, y) if epoch < self.warmup or self.two_stage else nll(mean, y, var)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            sched.step()
            net.eval()
            with torch.no_grad():
                mean, var = net(data["s_val"], data["p_val"])
                val_nll, val_mse = nll(mean, data["y_val"], var).item(), mse(mean, data["y_val"]).item()
            history.append([total / n, val_nll, val_mse])
            score = val_mse if self.two_stage else val_nll
            if (self.two_stage or epoch >= self.warmup or self.epochs <= self.warmup) and score < best:
                best, best_epoch = score, epoch
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        return net, np.array(history), best_epoch

    def _fit_spread(self, net, data):
        """two_stage: the network frozen, only the variance half of its last layer learns (Gaussian NLL)."""
        on = self._on()
        last = net.net[-1] if hasattr(net, "net") else net.out
        n_out = data["y_train"].shape[1]
        seen = {}
        hook = last.register_forward_hook(lambda module, inputs, output: seen.update(h=inputs[0]))
        net.eval()
        with torch.no_grad():
            mu_train, h_train = net(data["s_train"], data["p_train"])[0], seen["h"]
            mu_val, h_val = net(data["s_val"], data["p_val"])[0], seen["h"]
        hook.remove()
        head = nn.Linear(h_train.shape[1], n_out).to(on)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.log(torch.expm1(((mu_train - data["y_train"]) ** 2).mean(0))))
        opt = torch.optim.Adam(head.parameters(), lr=self.spread_lr)
        nll = nn.GaussianNLLLoss()
        variance = lambda h: nn.functional.softplus(head(h)) + 1e-6
        best, best_state = float("inf"), None
        for epoch in range(self.spread_epochs):
            order = torch.randperm(len(h_train), device=on)
            for start in range(0, len(h_train), self.batch_size):
                idx = order[start:start + self.batch_size]
                loss = nll(mu_train[idx], data["y_train"][idx], variance(h_train[idx]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                val_nll = nll(mu_val, data["y_val"], variance(h_val)).item()
            if val_nll < best:
                best, best_state = val_nll, {k: v.clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)
        with torch.no_grad():
            last.weight[n_out:] = head.weight
            last.bias[n_out:] = head.bias

    @torch.no_grad()
    def predict_with_spread(self, X):
        """(ensemble mean, sqrt(mean of the members' variances + variance of their means)), on Y's scale."""
        s, p = self._split_inputs(X)
        s = torch.tensor(self.s_scaler.transform(s), dtype=torch.float32)
        p = torch.tensor(self.p_scaler.transform(p), dtype=torch.float32)
        outs = [net.to("cpu").eval()(s, p) for net in self.members]
        mus = np.array([o[0].numpy() for o in outs]) * self.y_scaler.scale_ + self.y_scaler.mean_
        variances = np.array([o[1].numpy() for o in outs]) * self.y_scaler.scale_ ** 2
        spread = np.sqrt(variances.mean(0) + mus.var(0))
        scale = getattr(self, "spread_scale", None)
        return mus.mean(0), spread if scale is None else spread * scale

    def predict(self, X):
        return self.predict_with_spread(X)[0]

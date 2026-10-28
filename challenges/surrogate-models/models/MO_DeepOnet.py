"""MO_DeepOnet: a multiple-output DeepONet (DeepXDE, PyTorch backend) for the trajectory case.

The trajectory read as an operator: the branch net takes the machine settings (P_max, cN2_max,
L_inj, dip_frac, x_of), the trunk net takes the coordinate z_mm, and each of the six outputs is
the dot product of their features plus a bias. With DeepXDE's "split_branch" strategy (the
default) the trunk is shared: it ends in `latent` neurons, one set of shapes along z for every
output, and the branch ends in num_outputs x latent coefficients, one group per output.
"split_both" gives every output its own trunk features too. coordinate_net="branch" swaps
the two inputs (z_mm into the branch, the settings into the trunk).

Options beyond the plain DeepONet:
- coordinate_with: settings also given to the coordinate net, next to z_mm. ["l_inj"] lets the
  trunk place the plasma end (L_inj + 3.2 mm, a different z for every configuration), where the
  bunch length and the emittance change sharply.
- lr_decay="cosine": the learning rate falls from lr to lr_min over the iterations.
- ensemble=N: N nets trained with seeds seed, seed+1, ..., predictions averaged.
- baseline="linear_z": a ridge model, linear in the settings with coefficients that are a
  polynomial of degree baseline_degree in z, is fitted first; the nets learn only its residual.
  The linear part extrapolates smoothly beyond the training range (on log targets and a log
  P_max input it is a power law in density).
- weight_decay > 0: L2 on the weights (AdamW), a smoother net outside the training range.

Inputs and targets are standardised inside the model, so it trains the same whichever
normalisation the feature view applies. It needs the input column names to find z_mm
(accepts_dataframe), so it is for the trajectory case, including SPLIT = "ood_density".
"""

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde  # noqa: E402  (the backend is chosen at import)
import numpy as np  # noqa: E402
import torch  # noqa: E402

PARAMS = {"branch_layers": [128, 128, 128], "trunk_layers": [128, 128, 128], "latent": 32,
          "activation": "tanh", "strategy": "split_branch", "coordinate_net": "trunk",
          "coordinate_with": [], "iterations": 40000, "batch_size": 4096, "lr": 1e-3,
          "lr_decay": "cosine", "lr_min": 1e-5, "seed": 0, "ensemble": 1, "baseline": None,
          "baseline_degree": 3, "baseline_alpha": 1e-3, "weight_decay": 0.0, "verbose": 0}
# Chosen on the in-range holdout of the OOD split (never on the above-cut one): split_branch +
# cosine decay over 40k iterations, in_range 0.983 / above_cut 0.952. Tried and not kept:
# split_both (0.980 / 0.954, v1 of the registered model), L_inj in the trunk (0.982 / 0.950),
# silu (0.980 / 0.873), latent 64 (0.981 / 0.934).
# Best so far, on top of these defaults: {"ensemble": 5, "baseline": "linear_z"} with log10 P_max
# (feature view grl_trajectory_fv v2) -- registered grl_trajectory_mo_deeponet_ood v4: holdout
# 0.985 / 0.966, official (hidden test) 0.984 in distribution / 0.965 out, sigz_um 0.904.
STRATEGIES = ("split_both", "split_branch", "split_trunk", "independent")


def build_model(**params):
    return MODeepONet(**{**PARAMS, **params})


class MODeepONet:
    accepts_dataframe = True

    def __init__(self, branch_layers=(128, 128, 128), trunk_layers=(128, 128, 128), latent=32,
                 activation="tanh", strategy="split_both", coordinate_net="trunk",
                 coordinate_with=(), iterations=20000, batch_size=4096, lr=1e-3,
                 lr_decay=None, lr_min=1e-5, seed=0, ensemble=1, baseline=None,
                 baseline_degree=3, baseline_alpha=1e-3, weight_decay=0.0, verbose=0):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if coordinate_net not in ("trunk", "branch"):
            raise ValueError("coordinate_net must be 'trunk' or 'branch'")
        if lr_decay not in (None, "cosine"):
            raise ValueError("lr_decay must be None or 'cosine'")
        if baseline not in (None, "linear_z"):
            raise ValueError("baseline must be None or 'linear_z'")
        self.branch_layers, self.trunk_layers = list(branch_layers), list(trunk_layers)
        self.latent, self.activation, self.strategy = latent, activation, strategy
        self.coordinate_net, self.coordinate_with = coordinate_net, list(coordinate_with)
        self.iterations, self.batch_size, self.lr = iterations, batch_size, lr
        self.lr_decay, self.lr_min = lr_decay, lr_min
        self.seed, self.ensemble, self.verbose = seed, ensemble, verbose
        self.baseline, self.baseline_degree = baseline, baseline_degree
        self.baseline_alpha, self.weight_decay = baseline_alpha, weight_decay
        self.history = []

    def fit(self, X, Y):
        coordinate = [c for c in X.columns if "z_mm" in c]
        if len(coordinate) != 1:
            raise ValueError(f"MO_DeepOnet needs one z_mm input column, got {list(X.columns)}")
        self.coordinate = coordinate[0]
        self.settings = [c for c in X.columns if c != self.coordinate]
        self.coordinate_extra = [i for i, c in enumerate(self.settings)
                                 if any(name in c for name in self.coordinate_with)]
        if len(self.coordinate_extra) != len(self.coordinate_with):
            raise ValueError(f"coordinate_with {self.coordinate_with} not all in {self.settings}")
        settings, z = self._split(X)
        self.s_mean, self.s_std = settings.mean(0), settings.std(0) + 1e-8
        self.z_mean, self.z_std = z.mean(0), z.std(0) + 1e-8
        Y = np.asarray(Y, np.float32)
        self.y_mean, self.y_std = Y.mean(0), Y.std(0) + 1e-8
        self.n_outputs = Y.shape[1]

        inputs = self._net_inputs(settings, z)
        y = (Y - self.y_mean) / self.y_std
        self.baseline_coef = None
        if self.baseline == "linear_z":
            design = self._baseline_design(settings, z)
            gram = design.T @ design + self.baseline_alpha * len(y) * np.eye(design.shape[1])
            self.baseline_coef = np.linalg.solve(gram, design.T @ y).astype(np.float32)
            y = y - design @ self.baseline_coef
        self.r_std = y.std(0) + 1e-8                   # the nets see unit-variance targets
        y = (y / self.r_std).astype(np.float32)
        check = np.random.default_rng(self.seed).choice(len(y), min(len(y), 20000), replace=False)
        data = dde.data.Triple(X_train=inputs, y_train=y,
                               X_test=tuple(part[check] for part in inputs), y_test=y[check])
        decay = ("cosine", self.iterations, self.lr_min) if self.lr_decay == "cosine" else None
        self.nets, self.histories = [], []
        for member in range(self.ensemble):
            dde.config.set_random_seed(self.seed + member)
            net = self._build_net()
            model = dde.Model(data, net)
            model.compile("adamw" if self.weight_decay > 0 else "adam", lr=self.lr,
                          decay=decay, verbose=self.verbose)
            history, _ = model.train(iterations=self.iterations, batch_size=self.batch_size,
                                     display_every=max(1, self.iterations // 20),
                                     verbose=self.verbose)
            self.nets.append(net)
            self.histories.append([float(np.sum(loss)) for loss in history.loss_train])
        self.history = self.histories[0]
        return self

    @torch.no_grad()
    def predict(self, X):
        """The mean of the members' predictions (one member unless ensemble > 1)."""
        on = next(self.nets[0].parameters()).device
        inputs = tuple(torch.from_numpy(part).to(on) for part in self._net_inputs(*self._split(X)))
        out = np.mean([net.eval()(inputs).cpu().numpy() for net in self.nets], axis=0)
        out = out * self.r_std
        if self.baseline_coef is not None:
            out = out + self._baseline_design(*self._split(X)) @ self.baseline_coef
        return out * self.y_std + self.y_mean

    def _baseline_design(self, settings, z):
        """[1, settings] x [1, z, ..., z^degree]: linear in the settings, smooth in z."""
        s = (settings - self.s_mean) / self.s_std
        t = (z - self.z_mean) / self.z_std
        S = np.concatenate([np.ones((len(s), 1)), s], axis=1)
        Z = np.concatenate([t ** k for k in range(self.baseline_degree + 1)], axis=1)
        return (S[:, :, None] * Z[:, None, :]).reshape(len(s), -1).astype(np.float32)

    def _split(self, X):
        return (X[self.settings].to_numpy(np.float32), X[[self.coordinate]].to_numpy(np.float32))

    def _net_inputs(self, settings, z):
        settings = ((settings - self.s_mean) / self.s_std).astype(np.float32)
        z = ((z - self.z_mean) / self.z_std).astype(np.float32)
        coordinate = np.concatenate([z, settings[:, self.coordinate_extra]], axis=1)
        return (settings, coordinate) if self.coordinate_net == "trunk" else (coordinate, settings)

    def _build_net(self):
        n = self.n_outputs
        split_branch = self.strategy in ("split_both", "split_branch")
        split_trunk = self.strategy in ("split_both", "split_trunk")
        coordinate_in = 1 + len(self.coordinate_extra)
        branch_in, trunk_in = ((len(self.settings), coordinate_in) if self.coordinate_net == "trunk"
                               else (coordinate_in, len(self.settings)))
        return dde.nn.DeepONet(
            [branch_in] + self.branch_layers + [self.latent * (n if split_branch else 1)],
            [trunk_in] + self.trunk_layers + [self.latent * (n if split_trunk else 1)],
            self.activation, "Glorot normal", num_outputs=n, multi_output_strategy=self.strategy,
            regularization=["l2", self.weight_decay] if self.weight_decay > 0 else None)

    # DeepXDE objects do not pickle: save the weights, rebuild the nets on load
    def __getstate__(self):
        state = self.__dict__.copy()
        nets = state.pop("nets", [])
        state["nets_state"] = [{k: v.detach().cpu() for k, v in net.state_dict().items()}
                               for net in nets]
        return state

    def __setstate__(self, state):
        weights = state.pop("nets_state", None)
        if weights is None:                           # models saved with a single net (v1, v2)
            single = state.pop("net_state", None)
            weights = [single] if single is not None else []
        state.setdefault("ensemble", len(weights) or 1)
        state.setdefault("baseline", None)            # models saved before these options
        state.setdefault("baseline_coef", None)
        state.setdefault("weight_decay", 0.0)
        state.setdefault("r_std", np.ones_like(state["y_std"]))
        state.setdefault("coordinate_with", [])      # models saved before these options
        state.setdefault("coordinate_extra", [])
        state.setdefault("lr_decay", None)
        state.setdefault("lr_min", 1e-5)
        self.__dict__.update(state)
        self.nets = []
        for member in weights:
            net = self._build_net()
            net.load_state_dict(member)
            self.nets.append(net)

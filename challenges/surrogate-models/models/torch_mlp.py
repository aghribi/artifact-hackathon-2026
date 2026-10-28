"""A PyTorch multilayer perceptron with the same fit/predict contract; uses CUDA or Apple MPS if present."""

import numpy as np
import torch
from torch import nn

PARAMS = {"hidden": 256, "depth": 4, "epochs": 60, "batch_size": 4096, "lr": 2e-3, "seed": 0}


def build_model(**params):
    return TorchMLP(**{**PARAMS, **params})


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchMLP:
    def __init__(self, hidden=256, depth=4, epochs=60, batch_size=4096, lr=2e-3, seed=0):
        self.hidden, self.depth, self.epochs = hidden, depth, epochs
        self.batch_size, self.lr, self.seed = batch_size, lr, seed
        self.history = []

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-8
        self.y_mean, self.y_std = Y.mean(0), Y.std(0) + 1e-8
        layers, width = [], X.shape[1]
        for _ in range(self.depth):
            layers += [nn.Linear(width, self.hidden), nn.SiLU()]
            width = self.hidden
        self.net = nn.Sequential(*layers, nn.Linear(width, Y.shape[1]))

        on = device()
        self.net.to(on)
        x = torch.from_numpy((X - self.x_mean) / self.x_std).to(on)
        y = torch.from_numpy((Y - self.y_mean) / self.y_std).to(on)
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, self.epochs)
        for _ in range(self.epochs):
            order, total = torch.randperm(len(x), device=on), 0.0
            for start in range(0, len(x), self.batch_size):
                batch = order[start:start + self.batch_size]
                loss = nn.functional.mse_loss(self.net(x[batch]), y[batch])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += loss.item() * len(batch)
            schedule.step()
            self.history.append(total / len(x))
        self.net.to("cpu")   # saved and reloaded on any machine
        return self

    @torch.no_grad()
    def predict(self, X):
        x = torch.from_numpy((np.asarray(X, np.float32) - self.x_mean) / self.x_std)
        return self.net.eval()(x).numpy() * self.y_std + self.y_mean

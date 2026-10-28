"""A scikit-learn multilayer perceptron, all targets at once. CPU only; minutes on the trajectory."""

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARAMS = {"hidden": (128, 128, 128), "max_iter": 200, "seed": 0}


def build_model(hidden=(128, 128, 128), max_iter=200, seed=0):
    # the targets are standardised too: they come in on the scorer's scale, not unit variance
    from sklearn.compose import TransformedTargetRegressor
    net = MLPRegressor(hidden_layer_sizes=tuple(hidden), max_iter=max_iter, early_stopping=True,
                       random_state=seed)
    return TransformedTargetRegressor(make_pipeline(StandardScaler(), net),
                                      transformer=StandardScaler())

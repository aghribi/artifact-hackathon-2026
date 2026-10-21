"""Linear regression on the (optionally polynomial) inputs: the bar the Easy objectives ask you to beat.

A model file defines build_model(**params), returning an object with fit(X, Y) and predict(X)
on numpy arrays (Y with one column per target), and PARAMS, the defaults the notebook shows.
"""

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

PARAMS = {"degree": 1, "alpha": 1e-6}


def build_model(degree=1, alpha=1e-6):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(),
                         Ridge(alpha=alpha))

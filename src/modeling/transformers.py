"""Fit-in-fold transformers for logistic regression (use inside a Pipeline so CV stays leak-free)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Quantile-bin selected columns and replace them with smoothed weight of evidence.

    WoE_bin = log( P(bin | quit) / P(bin | stay) ), with additive smoothing `alpha` so sparse bins
    do not blow up. Bin edges and WoE values are learned from the training fold only.
    Expects and returns a DataFrame.
    """

    def __init__(self, cols=None, q: int = 6, alpha: float = 1.0, keep_raw: bool = False):
        self.cols, self.q, self.alpha, self.keep_raw = cols, q, alpha, keep_raw

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        n1, n0 = y.sum(), (1 - y).sum()
        self.maps_ = {}
        for c in self.cols:
            edges = np.unique(np.quantile(X[c], np.linspace(0, 1, self.q + 1)[1:-1]))
            b = np.digitize(X[c], edges)
            tab = {}
            for k in np.unique(b):
                m = b == k
                a1 = y[m].sum() + self.alpha
                a0 = (1 - y[m]).sum() + self.alpha
                tab[k] = np.log((a1 / (n1 + self.alpha * 2)) / (a0 / (n0 + self.alpha * 2)))
            self.maps_[c] = (edges, tab)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, (edges, tab) in self.maps_.items():
            woe = np.array([tab.get(k, 0.0) for k in np.digitize(X[c], edges)])
            if self.keep_raw:
                out[c + "_woe"] = woe
            else:
                out[c] = woe
        return out


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap columns at fold-learned percentiles."""

    def __init__(self, lo: float = 1, hi: float = 99):
        self.lo, self.hi = lo, hi

    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.percentile(X, self.lo, axis=0), np.percentile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X), self.lo_, self.hi_)

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from src.modeling import load_split, linear_features, pruned_linear_features, best_cv_scores, SEED

fs = load_split(); Xtr, ytr, Xte, yte = fs.X_train, fs.y_train, fs.X_test, fs.y_test
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
lin = linear_features(Xtr); pr = pruned_linear_features(Xtr, verbose=False)
cont = [c for c in pr if Xtr[c].nunique() > 10]
top = ["Overtime","LogIncome","Age","TenureRatio","AvgYearsPerCompany","MaritalStatus_Single","YearsSinceLastPromotion","DistanceFromHome"]

def LR(): return LogisticRegression(solver="saga", max_iter=20000)
grid = {"logisticregression__C": np.logspace(-2.5, 0, 7), "logisticregression__l1_ratio": [0.1, 0.5, 1.0]}
res = []
def run(name, Xa, Xb, pipe=None, g=grid):
    pipe = pipe or make_pipeline(StandardScaler(), LR())
    s = GridSearchCV(pipe, g, cv=cv, scoring="roc_auc", n_jobs=2).fit(Xa, ytr)
    sc = best_cv_scores(s); t = roc_auc_score(yte, s.predict_proba(Xb)[:, 1])
    res.append((name, sc.mean(), sc.std(), t, Xa.shape[1])); print(res[-1], flush=True)

run("A baseline EN (47 feats)", Xtr[lin], Xte[lin])

def poly(X):
    X = X.copy()
    for c in ["Age","LogIncome","TenureRatio","AvgYearsPerCompany","YearsSinceLastPromotion","DistanceFromHome","MostRecentPerfEval"]:
        X[c+"_sq"] = (X[c]-Xtr[c].mean())**2
    X["log_dist"] = np.log1p(X["DistanceFromHome"]); X["log_promo"] = np.log1p(X["YearsSinceLastPromotion"])
    X["log_age"] = np.log(X["Age"]); X["inv_tenure"] = 1/(1+X["AvgYearsPerCompany"])
    return X
run("B + squares/log terms", poly(Xtr[lin]), poly(Xte[lin]))

def inter(X):
    X = X.copy(); T = X[top]
    for i, a in enumerate(top):
        for b in top[i+1:]:
            X[f"{a}*{b}"] = T[a]*T[b]
    return X
run("C + pairwise interactions (top 8)", inter(Xtr[lin]), inter(Xte[lin]))

class LeafGBM(BaseEstimator, TransformerMixin):
    """Fit a small LightGBM inside each fold; one-hot its leaf indices as new features."""
    def fit(self, X, y):
        self.g = LGBMClassifier(n_estimators=30, num_leaves=4, max_depth=3, learning_rate=0.1, subsample=0.7,
                                subsample_freq=1, colsample_bytree=0.6, min_child_samples=20, random_state=SEED, verbose=-1, n_jobs=1).fit(X, y)
        self.oh = OneHotEncoder(handle_unknown="ignore").fit(self.g.predict(X, pred_leaf=True)); return self
    def transform(self, X):
        L = self.oh.transform(self.g.predict(X, pred_leaf=True)).toarray()
        return np.hstack([np.asarray(X), L])
gbm_pipe = Pipeline([("leaf", LeafGBM()), ("sc", StandardScaler()), ("logisticregression", LR())])
run("D + GBM leaf features (in-fold)", Xtr[pr], Xte[pr], gbm_pipe,
    {"logisticregression__C": np.logspace(-3, -0.5, 6), "logisticregression__l1_ratio": [0.1, 0.5, 1.0]})

class WoE(BaseEstimator, TransformerMixin):
    """Quantile-bin continuous columns in-fold, replace with smoothed weight of evidence."""
    def __init__(self, cols=None, q=6): self.cols, self.q = cols, q
    def fit(self, X, y):
        X = pd.DataFrame(X); y = np.asarray(y); self.maps = {}
        p0 = y.mean()
        for c in self.cols:
            edges = np.unique(np.quantile(X[c], np.linspace(0, 1, self.q+1)[1:-1]))
            b = np.digitize(X[c], edges); tab = {}
            for k in np.unique(b):
                m = b == k; n1 = y[m].sum() + 2*p0; n0 = (~(y[m]==1)).sum() + 2*(1-p0)
                tab[k] = np.log((n1/(y.sum()+2*p0)) / (n0/((1-y).sum()+2*(1-p0))))
            self.maps[c] = (edges, tab)
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c, (edges, tab) in self.maps.items():
            X[c+"_woe"] = [tab.get(k, 0.0) for k in np.digitize(X[c], edges)]
        return X.to_numpy()
cols_idx = list(pr)
woe_pipe = Pipeline([("woe", WoE(cols=cont)), ("sc", StandardScaler()), ("logisticregression", LR())])
class DF(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return pd.DataFrame(X, columns=cols_idx)
woe_pipe = Pipeline([("df", DF()), ("woe", WoE(cols=cont)), ("sc", StandardScaler()), ("logisticregression", LR())])
run("E + WoE of continuous (in-fold)", Xtr[pr], Xte[pr], woe_pipe)

out = pd.DataFrame(res, columns=["variant","cv_auc","cv_std","test_auc","n_in"]).round(4)
print(out.to_string(index=False)); out.to_csv("results/logit_boost_experiments.csv", index=False)

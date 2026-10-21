"""Round 3: prioritized logistic-regression boosts. Run from project root: python -m scripts.logit_boost_experiments3
Selection rule: pick by CV AUC (training data only). Test AUC is reported, never used to choose.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

from src.modeling import (SEED, WoEEncoder, best_cv_scores, linear_features, load_split,
                          pruned_linear_features)

fs = load_split(); Xtr, ytr, Xte, yte = fs.X_train, fs.y_train, fs.X_test, fs.y_test
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
lin, pr = linear_features(Xtr), pruned_linear_features(Xtr, verbose=False)
cont = lambda cols: [c for c in cols if Xtr[c].nunique() > 10]
LR = lambda **kw: LogisticRegression(solver="saga", max_iter=20000, **kw)
EN = {"lr__C": np.logspace(-2.5, 0, 6), "lr__l1_ratio": [0.1, 0.5, 1.0]}
rows, store = [], {}

def run(name, X, Xt, pipe, grid):
    s = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=2).fit(X, ytr)
    sc = best_cv_scores(s); p = s.predict_proba(Xt)[:, 1]
    rows.append(dict(variant=name, cv_auc=sc.mean(), cv_std=sc.std(), test_auc=roc_auc_score(yte, p),
                     best={k.split("__")[-1]: round(float(v), 3) for k, v in s.best_params_.items()}))
    store[name] = p; print(rows[-1], flush=True)

# 0. reference
run("0 baseline EN (47 feats)", Xtr[lin], Xte[lin], Pipeline([("sc", StandardScaler()), ("lr", LR())]), EN)

# 1. WoE, tune bins + smoothing, pruned set (highest priority: monotone-free, robust to outliers, few params)
woe = lambda cols, **kw: Pipeline([("woe", WoEEncoder(cols=cont(cols), **kw)), ("sc", StandardScaler()), ("lr", LR())])
run("1 WoE pruned (q, alpha tuned)", Xtr[pr], Xte[pr], woe(pr), {**EN, "woe__q": [4, 6, 8, 10], "woe__alpha": [0.5, 2.0]})
# 2. WoE on the full 47 set (let elastic net handle collinearity instead of VIF pruning)
run("2 WoE full-47 + EN", Xtr[lin], Xte[lin], woe(lin), {**EN, "woe__q": [4, 6, 8], "woe__alpha": [1.0]})
# 3. keep raw + add WoE copy
run("3 raw + WoE copy (pruned)", Xtr[pr], Xte[pr], woe(pr, keep_raw=True), {**EN, "woe__q": [4, 6, 8], "woe__alpha": [1.0]})

# 4. Splines with penalty (GAM-like), EN penalty instead of L2 only
sp_cols, ln_cols = cont(pr), [c for c in pr if c not in cont(pr)]
spl = Pipeline([("pre", ColumnTransformer([("s", SplineTransformer(degree=3, n_knots=4, knots="quantile", extrapolation="linear"), sp_cols),
                                           ("l", "passthrough", ln_cols)])), ("sc", StandardScaler()), ("lr", LR())])
run("4 splines + EN (pruned)", Xtr[pr], Xte[pr], spl, {**EN, "pre__s__n_knots": [3, 4, 5]})

# 5. Bagged WoE logistic (variance reduction; ranking averaged over bootstraps)
bag = Pipeline([("woe", WoEEncoder(cols=cont(pr), q=6)), ("sc", StandardScaler()),
                ("lr", BaggingClassifier(LR(C=0.1, l1_ratio=0.1), n_estimators=40, max_samples=0.8, random_state=SEED, n_jobs=1))])
run("5 bagged WoE logistic", Xtr[pr], Xte[pr], bag, {"woe__q": [4, 6, 8]})

# 6/7. Resampling inside each fold (imblearn pipeline); expected NOT to help AUC
for nm, smp in [("6 SMOTE", SMOTE(random_state=SEED)), ("7 undersample", RandomUnderSampler(random_state=SEED))]:
    p = ImbPipeline([("woe", WoEEncoder(cols=cont(pr), q=6)), ("sc", StandardScaler()), ("smp", smp), ("lr", LR())])
    run(f"{nm} + WoE", Xtr[pr], Xte[pr], p, EN)

out = pd.DataFrame(rows).round(4)
out.to_csv("results/logit_boost_experiments3.csv", index=False)
pd.DataFrame(store).assign(y=yte.values).to_csv("results/preds/logit_boost3_test_scores.csv", index=False)

# paired bootstrap of test AUC difference vs baseline (how much of the gap is noise?)
rng = np.random.default_rng(SEED); y = yte.values; base = store["0 baseline EN (47 feats)"]
idx = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
idx = [i for i in idx if 0 < y[i].sum() < len(i)]
print("\nvariant | test AUC | diff vs baseline | 95% bootstrap CI of diff")
for k, p in store.items():
    d = np.array([roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], base[i]) for i in idx])
    print(f"{k:34s} {roc_auc_score(y, p):.4f}  {roc_auc_score(y, p) - roc_auc_score(y, base):+.4f}  [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
print("\n", out.drop(columns="best").to_string(index=False))

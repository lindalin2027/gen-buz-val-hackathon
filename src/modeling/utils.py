"""Shared helpers for the model notebooks: data loading, CV, VIF pruning, scoring and logging."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

from src.data_cleaning import build_features

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "EmployeeAttritionPredictionData.xlsx"
RESULTS_DIR = ROOT / "results"
SCORES_PATH = RESULTS_DIR / "model_scores.csv"

ENGINEERED = [
    "TenureRatio", "RoleRatio", "MgrRatio", "PromoGap", "StagnationYears", "AvgYearsPerCompany",
    "JobHopper", "EarlyCareer", "NewHire", "AgeAtJoin", "LogIncome", "IncomePerLevel",
    "IncomePerYearWork", "IncomePerAge", "OT_x_Travel", "OT_x_Single", "FarFromHome", "LowPerf",
    "TravelOrd", "Male",
]
# Reference level dropped per nominal group (avoids the dummy trap in linear models).
REFERENCE_DUMMIES = [
    "Department_Research & Development",
    "JobRole_Research Scientist",
    "MaritalStatus_Married",
    "BusinessTravel_Travel_Rarely",
    "Gender_Female",
    "Gender_Male",          # redundant with the engineered `Male` flag
]


# Hand pruning for linear models, decided in EDA:
# exact linear combinations (AgeAtJoin, StagnationYears, travel dummies vs TravelOrd),
# income/level redundancy (keep LogIncome), and Department (nested in JobRole).
HAND_DROP = [
    "MonthlyIncome", "JobLevel", "IncomePerAge", "IncomePerLevel",
    "AgeAtJoin", "StagnationYears",
    "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently",
    "Department_Human Resources", "Department_Sales",
]


def load_split():
    """Return the FeatureSet (X_train, y_train, X_test, y_test, id_train, id_test)."""
    return build_features(DATA_PATH, verbose=False)


def get_cv(n_splits: int = 5, n_repeats: int = 5, seed: int = SEED) -> RepeatedStratifiedKFold:
    """Repeated stratified K-fold used by every notebook so CV AUCs are comparable."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def raw_features(X: pd.DataFrame) -> list[str]:
    """Columns that are not engineered (original numerics plus one-hot dummies)."""
    return [c for c in X.columns if c not in ENGINEERED]


def linear_features(X: pd.DataFrame) -> list[str]:
    """All columns minus one reference dummy per group (for linear models)."""
    return [c for c in X.columns if c not in REFERENCE_DUMMIES]


def pruned_linear_features(X: pd.DataFrame, threshold: float = 5.0, verbose: bool = True) -> list[str]:
    """Hand pruning + iterative VIF pruning (keeps LogIncome, Overtime, Age)."""
    cols = [c for c in linear_features(X) if c not in HAND_DROP]
    return vif_prune(X[cols], threshold=threshold, keep=["LogIncome", "Overtime", "Age"], verbose=verbose)


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor per column: 1 / (1 - R^2) from regressing it on the others."""
    Z = (X - X.mean()) / X.std(ddof=0)
    Z = Z.to_numpy()
    out = {}
    for j, col in enumerate(X.columns):
        y, others = Z[:, j], np.delete(Z, j, axis=1)
        A = np.column_stack([np.ones(len(y)), others])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        r2 = 1 - resid.var() / y.var()
        out[col] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return pd.Series(out).sort_values(ascending=False)


def vif_prune(X: pd.DataFrame, threshold: float = 5.0, keep: list[str] | None = None,
              verbose: bool = True) -> list[str]:
    """Iteratively drop the highest-VIF column until all VIFs <= threshold.

    Unsupervised (does not look at y), so running it on the full training set does not leak the target.
    Columns in `keep` are never dropped.
    """
    keep = set(keep or [])
    cols = list(X.columns)
    while True:
        v = vif(X[cols])
        candidates = v.drop(labels=[c for c in v.index if c in keep])
        if candidates.empty or candidates.iloc[0] <= threshold:
            break
        drop = candidates.index[0]
        if verbose:
            print(f"drop {drop:<35s} VIF={candidates.iloc[0]:.1f}")
        cols.remove(drop)
    return cols


def cv_summary(scores) -> str:
    scores = np.asarray(scores)
    return f"CV AUC {scores.mean():.4f} +/- {scores.std():.4f} ({len(scores)} folds)"


def best_cv_scores(search) -> np.ndarray:
    """Per-fold AUCs of the best candidate from a fitted GridSearchCV / RandomizedSearchCV."""
    res, i = search.cv_results_, search.best_index_
    n = search.n_splits_
    return np.array([res[f"split{k}_test_score"][i] for k in range(n)])


def oof_scores(estimator, X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> np.ndarray:
    """Out-of-fold predicted probabilities on the training set (one 5-fold pass), for blending later."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def finalize(name: str, model, X_train, y_train, X_test, y_test, id_train, id_test,
             cv_scores, notes: str = "") -> float:
    """Score the test set ONCE with an already-fitted model, save predictions, and log the result.

    Writes results/preds/<name>_test.csv, results/preds/<name>_oof.csv and upserts results/model_scores.csv.
    """
    (RESULTS_DIR / "preds").mkdir(parents=True, exist_ok=True)
    p_test = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, p_test)

    pd.DataFrame({"ID": id_test, "y": y_test, "score": p_test}).to_csv(
        RESULTS_DIR / "preds" / f"{name}_test.csv", index=False)
    pd.DataFrame({"ID": id_train, "y": y_train, "score": oof_scores(model, X_train, y_train)}).to_csv(
        RESULTS_DIR / "preds" / f"{name}_oof.csv", index=False)

    cv_scores = np.asarray(cv_scores)
    row = pd.DataFrame([{
        "model": name,
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
        "test_auc": round(test_auc, 4),
        "n_features": X_train.shape[1],
        "notes": notes,
    }])
    if SCORES_PATH.exists():
        old = pd.read_csv(SCORES_PATH)
        row = pd.concat([old[old["model"] != name], row], ignore_index=True)
    row.sort_values("cv_auc_mean", ascending=False).to_csv(SCORES_PATH, index=False)
    print(f"{name}: {cv_summary(cv_scores)} | TEST AUC {test_auc:.4f}")
    return test_auc


def load_scores() -> pd.DataFrame:
    return pd.read_csv(SCORES_PATH) if SCORES_PATH.exists() else pd.DataFrame()

"""Data cleaning and feature engineering for the Next Labs attrition hackathon.

All transformations are fit-free (no statistics are learned from the data), so the
same code can be applied to train and test rows without leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = "data/EmployeeAttritionPredictionData.xlsx"
TARGET_RAW = "Attrition"
TARGET = "y"
SPLIT_COL = "training"  # 1 = train, 0 = test
ID_COL = "ID"
NOMINAL_COLS = ["Department", "JobRole", "MaritalStatus", "BusinessTravel", "Gender"]


@dataclass
class FeatureSet:
    """Model-ready data. Test labels are for final scoring only."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    id_train: pd.Series
    id_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def load_data(path: str | Path = DEFAULT_DATA_PATH, sheet_name: str = "Data") -> pd.DataFrame:
    """Read the raw employee sheet from the Excel file."""
    return pd.read_excel(path, sheet_name=sheet_name)


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Basic cleaning: encode the target and run data quality checks.

    - Adds `y` (Attrition: Yes = 1, No = 0).
    - Raises if there are missing values or duplicate IDs.
    - Keeps MostRecentPerfEval == 0 rows (dictionary says 1 to 10); they are
      treated as low performers downstream, not dropped.
    """
    out = df.copy()

    if out.isna().any().any():
        raise ValueError("Missing values found; this pipeline assumes none.")
    if out[ID_COL].duplicated().any():
        raise ValueError("Duplicate employee IDs found.")
    out = out.drop_duplicates()

    out[TARGET] = (out[TARGET_RAW] == "Yes").astype(int)

    if verbose:
        n_bad_perf = int((~out["MostRecentPerfEval"].between(1, 10)).sum())
        if n_bad_perf:
            print(f"[clean] {n_bad_perf} rows have MostRecentPerfEval outside 1-10 (kept).")
        tenure_checks = {
            "YearsAtCompany > TotalWorkingYears": out["YearsAtCompany"] > out["TotalWorkingYears"],
            "YearsInCurrentRole > YearsAtCompany": out["YearsInCurrentRole"] > out["YearsAtCompany"],
            "YearsWithCurrManager > YearsAtCompany": out["YearsWithCurrManager"] > out["YearsAtCompany"],
            "YearsSinceLastPromotion > YearsAtCompany": out["YearsSinceLastPromotion"] > out["YearsAtCompany"],
            "TotalWorkingYears > Age - 14": out["TotalWorkingYears"] > out["Age"] - 14,
        }
        for name, bad in tenure_checks.items():
            if bad.any():
                print(f"[clean] WARNING: {int(bad.sum())} rows violate: {name}")
    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features (20 new columns). Denominators are guarded with +1."""
    x = df.copy()

    # Tenure and career ratios
    x["TenureRatio"] = x["YearsAtCompany"] / (x["TotalWorkingYears"] + 1)
    x["RoleRatio"] = x["YearsInCurrentRole"] / (x["YearsAtCompany"] + 1)
    x["MgrRatio"] = x["YearsWithCurrManager"] / (x["YearsAtCompany"] + 1)
    x["PromoGap"] = x["YearsSinceLastPromotion"] / (x["YearsAtCompany"] + 1)
    x["StagnationYears"] = x["YearsAtCompany"] - x["YearsSinceLastPromotion"]
    x["AvgYearsPerCompany"] = x["TotalWorkingYears"] / (x["NumCompaniesWorked"] + 1)
    x["JobHopper"] = (x["NumCompaniesWorked"] >= 5).astype(int)
    x["EarlyCareer"] = (x["TotalWorkingYears"] <= 3).astype(int)
    x["NewHire"] = (x["YearsAtCompany"] <= 2).astype(int)
    x["AgeAtJoin"] = x["Age"] - x["YearsAtCompany"]

    # Pay
    x["LogIncome"] = np.log1p(x["MonthlyIncome"])
    x["IncomePerLevel"] = x["MonthlyIncome"] / x["JobLevel"]
    x["IncomePerYearWork"] = x["MonthlyIncome"] / (x["TotalWorkingYears"] + 1)
    x["IncomePerAge"] = x["MonthlyIncome"] / x["Age"]

    # Workload and context
    x["OT_x_Travel"] = x["Overtime"] * (x["BusinessTravel"] == "Travel_Frequently").astype(int)
    x["OT_x_Single"] = x["Overtime"] * (x["MaritalStatus"] == "Single").astype(int)
    x["FarFromHome"] = (x["DistanceFromHome"] >= 20).astype(int)
    x["LowPerf"] = (x["MostRecentPerfEval"] <= 3).astype(int)

    # Deterministic encodings
    x["TravelOrd"] = x["BusinessTravel"].map({"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2})
    x["Male"] = (x["Gender"] == "Male").astype(int)
    return x


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal columns and drop the raw target column."""
    return pd.get_dummies(df.drop(columns=[TARGET_RAW]), columns=NOMINAL_COLS, dtype=int)


def build_features(path: str | Path = DEFAULT_DATA_PATH, verbose: bool = True) -> FeatureSet:
    """Full pipeline: load -> clean -> engineer -> encode -> split on `training`."""
    df = clean(load_data(path), verbose=verbose)
    all_x = encode(engineer(df))

    feature_cols = [c for c in all_x.columns if c not in (ID_COL, SPLIT_COL, TARGET)]
    is_train = all_x[SPLIT_COL] == 1

    fs = FeatureSet(
        X_train=all_x.loc[is_train, feature_cols].reset_index(drop=True),
        y_train=all_x.loc[is_train, TARGET].reset_index(drop=True),
        X_test=all_x.loc[~is_train, feature_cols].reset_index(drop=True),
        y_test=all_x.loc[~is_train, TARGET].reset_index(drop=True),
        id_train=all_x.loc[is_train, ID_COL].reset_index(drop=True),
        id_test=all_x.loc[~is_train, ID_COL].reset_index(drop=True),
    )
    if verbose:
        print(
            f"[build] train {fs.X_train.shape}, test {fs.X_test.shape}, "
            f"features {len(feature_cols)}, quit rate train {fs.y_train.mean():.3f} / test {fs.y_test.mean():.3f}"
        )
    return fs


def save_features(fs: FeatureSet, out_dir: str | Path = "data") -> tuple[Path, Path]:
    """Write train_features.csv and test_features.csv (features + y + ID)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = out_dir / "train_features.csv", out_dir / "test_features.csv"
    fs.X_train.assign(y=fs.y_train, ID=fs.id_train).to_csv(train_path, index=False)
    fs.X_test.assign(y=fs.y_test, ID=fs.id_test).to_csv(test_path, index=False)
    return train_path, test_path

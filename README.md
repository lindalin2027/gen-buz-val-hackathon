# Next Labs Employee Attrition Hackathon

Predict which employees quit, scored by AUC on a 300-person test set. See `CLAUDE.md` for the full case rules.

## Project layout

```
Hackathon/
├── data/
│   ├── EmployeeAttritionPredictionData.xlsx   # raw data (sheets: Data, DataDictionary)
│   ├── train_features.csv                     # generated
│   └── test_features.csv                      # generated
├── notebooks/
│   └── 01_eda.ipynb                           # exploration and feature ideas
├── src/
│   └── data_cleaning/                         # cleaning + feature engineering module
└── README.md
```

## `src/data_cleaning`

### What it does

`build_features()` runs the whole pipeline in one call:

1. **Load** the `Data` sheet from the Excel file.
2. **Clean**
   - Encodes the target: `Attrition` Yes/No to `y` 1/0.
   - Checks for missing values and duplicate IDs (raises if found).
   - Prints warnings for inconsistent tenure/age values and for `MostRecentPerfEval` outside 1 to 10 (8 rows, kept as low performers).
3. **Engineer 20 features**
   - Career: `AvgYearsPerCompany`, `JobHopper`, `AgeAtJoin`, `EarlyCareer`, `NewHire`
   - Tenure and promotion: `TenureRatio`, `RoleRatio`, `MgrRatio`, `PromoGap`, `StagnationYears`
   - Pay: `LogIncome`, `IncomePerLevel`, `IncomePerYearWork`, `IncomePerAge`
   - Context: `OT_x_Travel`, `OT_x_Single`, `FarFromHome`, `LowPerf`, `TravelOrd`, `Male`
4. **One-hot encode** `Department`, `JobRole`, `MaritalStatus`, `BusinessTravel`, `Gender`.
5. **Split** on the `training` flag (1 = train, 0 = test) and drop `ID`, `training` and the raw target from the features.

Result: 53 model-ready numeric columns. All transforms are fit-free (nothing is learned from the data), so train and test are treated identically with no leakage. Scaling, if needed, must be fit inside CV folds.

### Run it from the command line

From the project root (`Hackathon/`):

```bash
python -m src.data_cleaning                                  # defaults
python -m src.data_cleaning --data data/EmployeeAttritionPredictionData.xlsx --out data
```

This writes `data/train_features.csv` and `data/test_features.csv` (features plus `y` and `ID`).

### Import it in Python or Jupyter

From the project root:

```python
from src.data_cleaning import build_features

fs = build_features()          # returns a FeatureSet
fs.X_train, fs.y_train         # 1170 rows x 53 features
fs.X_test, fs.y_test           # 300 rows (use y_test only for final scoring)
fs.feature_names               # list of 53 column names
```

From a notebook inside `notebooks/`, add the project root to the path first:

```python
import sys
sys.path.append("..")
from src.data_cleaning import build_features
fs = build_features("../data/EmployeeAttritionPredictionData.xlsx")
```

### Public functions

| Function | Purpose |
|---|---|
| `load_data(path)` | Read the raw `Data` sheet |
| `clean(df)` | Encode target, run quality checks |
| `engineer(df)` | Add the 20 engineered features |
| `encode(df)` | One-hot encode nominal columns |
| `build_features(path)` | Full pipeline, returns `FeatureSet` |
| `save_features(fs, out_dir)` | Write the two CSV files |

## Setup

Python (version per `pyproject.toml`; the code also runs on 3.10) with `pandas`, `numpy` and `openpyxl` (declared in `pyproject.toml`):

```bash
uv sync        # or: pip install pandas numpy openpyxl
```

## Rules to remember

- Fit and tune on the 1,170 training rows only. Touch the test set only for final AUC.
- Output a continuous score or probability, never a hard yes/no.

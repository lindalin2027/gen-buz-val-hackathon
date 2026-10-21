# Hackathon: Predicting Employee Attrition at Next Labs

Course: Northwestern, Generating Business Value with DS (Prof. Joel Shapiro).
Case is not for distribution outside Kellogg. Do not duplicate or post.

## Files
- `Hackathon_Instructions.pdf`: case description and rules
- `EmployeeAttritionPredictionData.xlsx`: employee data (1,470 rows; includes train/test split and possibly a data dictionary, to be confirmed)

## Scenario
Freelance data scientist hired by Angie Li, CEO of Next Labs (mid-sized lab equipment and software manufacturer). She worries too many employees are quitting and wants a model that most accurately predicts who will quit.

## Data
- 1,470 employees: demographics, job role, business travel, performance, etc.
- Target: `attrition` (has the employee ever quit)
- Train: 1,170 rows, about 16% quit
- Test: 300 rows, 52 quit (about 17%)

## Rules
- Build the model using ONLY the 1,170 training records, then score the 300 test records.
- Do NOT change the train/test data.
- Scoring metric: AUC on the 300-person test set. Highest team AUC wins.
- Output must be a continuous score or probability per person. Hard yes/no predictions cannot be ranked and cannot win.
- Any tools or methods allowed; any number of iterations in the time allotted.
- Must understand and be able to explain the model and what it means. Extra diagnostics are encouraged.
- High AUC does not imply calibrated probabilities (AUC only measures ranking).
- Team project.

## Working conventions
- Language: R or Python, both allowed. Code is written in a Jupyter notebook (R kernel for R, Python kernel for Python).
- Code should be clean and copy-paste-ready, one logical step per cell.
- Keep explanations concise.
- Avoid em dashes in writing.
- Tune and cross-validate on training data only. Touch the test set only for final scoring.
- Use `set.seed()` for reproducibility.
- Keep a record of each model tried, its CV AUC, and test AUC, for the write-up.

## Suggested plan
1. Inspect data, data dictionary, and train/test indicator.
2. Baseline: logistic regression, report CV AUC.
3. Try regularized logistic (glmnet or scikit-learn), random forest, gradient boosting (xgboost or lightgbm); compare by CV AUC.
4. Consider ensembling (average of rank or probability scores).
5. Score the test set, compute AUC, export predictions.
6. Prepare an explanation: key drivers (coefficients or variable importance), how AUC is read, and caveats on calibration.

## Progress log
- `01_eda.ipynb` (Python): EDA, cleaning checks, feature engineering. Writes `train_features.csv` and `test_features.csv` (53 model-ready columns plus `y` and `ID`).
- Data facts: no missing values or duplicates; 8 rows with MostRecentPerfEval = 0 (dictionary says 1 to 10); `training` flag: 1 = train, 0 = test; test set shows no meaningful distribution shift.
- Best raw univariate AUC (train): TotalWorkingYears 0.68. Best engineered: AvgYearsPerCompany 0.71.
- Next: `02_modeling.ipynb` with stratified repeated CV on train only.
- `src/data_cleaning/` package wraps the EDA cleaning and feature engineering (`python -m src.data_cleaning` or `from src.data_cleaning import build_features`). Verified to reproduce `data/train_features.csv` and `data/test_features.csv` exactly. Layout: raw data and generated CSVs in `data/`, notebooks in `notebooks/`.
- Modeling (notebooks 02-06, `src/modeling/`): CV = repeated stratified 5x5 on train only; test scored once via `finalize()`. Leaderboard in `results/model_scores.csv`. Logistic family CV AUC about 0.815-0.824; trees about 0.80.
- Logistic boost experiments (`scripts/logit_boost_experiments*.py`, `results/logit_boost_experiments*.csv`): WoE binning (q=4, alpha=2) gave the best CV (0.8235 vs 0.8184 baseline); splines+EN best test (0.780). SMOTE, undersampling, class_weight hurt. Test AUC differences up to +0.015 have bootstrap CIs that include 0 (only 52 test quitters): select by CV, not by test AUC.

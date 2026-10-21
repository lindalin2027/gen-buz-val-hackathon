"""Evidence for the recommendations: EDA rates, model odds ratios, grouped permutation importance, interaction tests.
Run from project root: python -m scripts.insights   (training data only)"""
import warnings; warnings.filterwarnings("ignore")
import itertools
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from src.modeling import SEED, load_split, linear_features, vif, get_cv

fs = load_split(); X, y = fs.X_train.copy(), fs.y_train
base_rate = y.mean()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h

# ---------------------------------------------------------------- 1. EDA evidence
def rate(mask, label):
    n, k = int(mask.sum()), int(y[mask].sum()); lo, hi = wilson(k, n)
    return dict(group=label, n=n, quit_rate=k / n, ci_lo=lo, ci_hi=hi)
yrs = X["YearsAtCompany"]
eda = pd.DataFrame([
    rate(X.Overtime == 1, "Overtime: yes"), rate(X.Overtime == 0, "Overtime: no"),
    rate(X["BusinessTravel_Travel_Frequently"] == 1, "Travel: frequently"),
    rate(X["BusinessTravel_Travel_Rarely"] == 1, "Travel: rarely"), rate(X["BusinessTravel_Non-Travel"] == 1, "Travel: none"),
    rate(yrs <= 1, "Tenure: first year (<=1 yr)"), rate((yrs > 1) & (yrs <= 3), "Tenure: 2-3 yrs"), rate((yrs > 3) & (yrs <= 10), "Tenure: 4-10 yrs"), rate(yrs > 10, "Tenure: >10 yrs"),
    rate(X.MonthlyIncome <= 2400, "Income: lowest eighth (<=$2.4k)"), rate(X.MonthlyIncome >= 8400, "Income: >=$8.4k"),
    rate(X.JobLevel == 1, "Job level 1"), rate(X.JobLevel == 2, "Job level 2"), rate(X.JobLevel >= 4, "Job level 4-5"),
    rate(X["JobRole_Sales Representative"] == 1, "Role: Sales Representative"), rate(X["JobRole_Human Resources"] == 1, "Role: Human Resources"),
    rate(X["JobRole_Laboratory Technician"] == 1, "Role: Laboratory Technician"), rate(X["JobRole_Manager"] == 1, "Role: Manager"),
    rate(X["JobRole_Research Director"] == 1, "Role: Research Director"),
    rate(X["MaritalStatus_Single"] == 1, "Marital: single"), rate(X["MaritalStatus_Single"] == 0, "Marital: not single"),
    rate(X.DistanceFromHome >= 20, "Distance >= 20 mi"), rate(X.DistanceFromHome < 20, "Distance < 20 mi"),
    rate(X.MostRecentPerfEval <= 3, "Perf eval <= 3"), rate(X.MostRecentPerfEval >= 8, "Perf eval >= 8"),
    rate(X.NumCompaniesWorked >= 5, "Prior employers >= 5"), rate(X.NumCompaniesWorked <= 4, "Prior employers <= 4"),
])
eda["lift_vs_overall"] = eda.quit_rate / base_rate
print("== EDA quit rates (train, 95% Wilson CI) overall", round(base_rate, 3)); print(eda.round(3).to_string(index=False))
eda.to_csv("results/insights_eda_rates.csv", index=False)

# ---------------------------------------------------------------- 2. Explanatory logistic (odds ratios)
D = pd.DataFrame({
    "Overtime": X.Overtime,
    "Travel_frequently": X["BusinessTravel_Travel_Frequently"], "Travel_none": X["BusinessTravel_Non-Travel"],
    "Single": X["MaritalStatus_Single"], "Divorced": X["MaritalStatus_Divorced"],
    "LogMonthlyIncome": np.log(X.MonthlyIncome),
    "YearsAtCompany": X.YearsAtCompany, "TotalWorkingYears": X.TotalWorkingYears,
    "NumCompaniesWorked": X.NumCompaniesWorked, "YearsSinceLastPromotion": X.YearsSinceLastPromotion,
    "YearsWithCurrManager": X.YearsWithCurrManager, "DistanceFromHome_10mi": X.DistanceFromHome / 10,
    "MostRecentPerfEval": X.MostRecentPerfEval,
    "Role_SalesRep": X["JobRole_Sales Representative"], "Role_LabTech": X["JobRole_Laboratory Technician"],
    "Role_HR": X["JobRole_Human Resources"], "Role_Manager": X["JobRole_Manager"],
    "Role_ResearchDirector": X["JobRole_Research Director"], "Role_SalesExec": X["JobRole_Sales Executive"],
    "Role_HealthcareRep": X["JobRole_Healthcare Representative"], "Role_ManufDirector": X["JobRole_Manufacturing Director"],
})
print("\n== max VIF of explanatory model:", round(vif(D).max(), 1), "|", vif(D).head(3).round(1).to_dict())
cvauc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000)), D, y, cv=get_cv(), scoring="roc_auc", n_jobs=2)
print(f"explanatory logistic CV AUC {cvauc.mean():.4f} +/- {cvauc.std():.4f}")
m = sm.Logit(y, sm.add_constant(D)).fit(disp=0)
ci = m.conf_int()
orr = pd.DataFrame({"odds_ratio": np.exp(m.params), "ci_lo": np.exp(ci[0]), "ci_hi": np.exp(ci[1]), "p": m.pvalues}).drop("const")
# income expressed per +20%
f = np.log(1.2); orr.loc["LogMonthlyIncome"] = [np.exp(m.params["LogMonthlyIncome"] * f), np.exp(ci.loc["LogMonthlyIncome", 0] * f), np.exp(ci.loc["LogMonthlyIncome", 1] * f), m.pvalues["LogMonthlyIncome"]]
orr = orr.rename(index={"LogMonthlyIncome": "Income +20% (per)"})
print("\n== odds ratios (OR>1 = more likely to quit; roles vs Research Scientist; marital vs married; travel vs rarely)")
print(orr.sort_values("odds_ratio", ascending=False).round(3).to_string())
orr.to_csv("results/insights_odds_ratios.csv")

# average marginal effects by setting a variable for everyone (g-computation)
def ame(col, v0, v1, extra=None):
    a, b = sm.add_constant(D).copy(), sm.add_constant(D).copy()
    a[col], b[col] = v0, v1
    if extra: 
        for k, (x0, x1) in extra.items(): a[k], b[k] = x0, x1
    return m.predict(a).mean(), m.predict(b).mean()
me = {}
me["Overtime: no -> yes"] = ame("Overtime", 0, 1)
me["Travel: rarely -> frequently"] = ame("Travel_frequently", 0, 1)
me["Single vs married"] = ame("Single", 0, 1, {"Divorced": (0, 0)})
me["Income +20%"] = (m.predict(sm.add_constant(D)).mean(), m.predict(sm.add_constant(D).assign(LogMonthlyIncome=D.LogMonthlyIncome + np.log(1.2))).mean())
me["Years at company: 1 -> 5"] = ame("YearsAtCompany", 1, 5)
me["Role: Sales Rep (vs Research Scientist)"] = ame("Role_SalesRep", 0, 1)
me["Distance: +10 miles"] = (m.predict(sm.add_constant(D)).mean(), m.predict(sm.add_constant(D).assign(DistanceFromHome_10mi=D.DistanceFromHome_10mi + 1)).mean())
mem = pd.DataFrame(me, index=["avg_pred_prob_before", "avg_pred_prob_after"]).T
mem["change_pts"] = (mem.avg_pred_prob_after - mem.avg_pred_prob_before) * 100
print("\n== average predicted quit probability if we change one thing for everyone, others held fixed"); print(mem.round(3).to_string())
mem.to_csv("results/insights_marginal_effects.csv")

# ---------------------------------------------------------------- 3. Grouped permutation importance (held-out folds)
groups = {
    "Overtime": ["Overtime", "OT_x_Travel", "OT_x_Single"],
    "Business travel": ["TravelOrd", "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely"],
    "Job role": [c for c in X if c.startswith("JobRole_") or c.startswith("Department_")],
    "Pay and level": ["MonthlyIncome", "LogIncome", "JobLevel", "IncomePerLevel", "IncomePerYearWork", "IncomePerAge"],
    "Tenure at company": ["YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager", "TenureRatio", "RoleRatio", "MgrRatio", "NewHire", "StagnationYears", "AgeAtJoin"],
    "Career length / age": ["TotalWorkingYears", "Age", "EarlyCareer"],
    "Job hopping": ["NumCompaniesWorked", "AvgYearsPerCompany", "JobHopper"],
    "Marital status": [c for c in X if c.startswith("MaritalStatus_")],
    "Promotion gap": ["YearsSinceLastPromotion", "PromoGap"],
    "Distance from home": ["DistanceFromHome", "FarFromHome"],
    "Performance eval": ["MostRecentPerfEval", "LowPerf"],
    "Education / gender": ["Education", "Male", "Gender_Female", "Gender_Male"],
}
lin = linear_features(X)
groups = {k: [c for c in v if c in lin] for k, v in groups.items()}
rng = np.random.default_rng(SEED); res = {k: [] for k in groups}
for rep in range(3):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(X, y):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="saga", C=0.1, l1_ratio=0.1, max_iter=20000)).fit(X.iloc[tr][lin], y.iloc[tr])
        Xte, yte = X.iloc[te][lin], y.iloc[te]; base = roc_auc_score(yte, mdl.predict_proba(Xte)[:, 1])
        for g, cols in groups.items():
            drops = []
            for _ in range(5):
                Xp = Xte.copy(); idx = rng.permutation(len(Xp)); Xp[cols] = Xp[cols].to_numpy()[idx]
                drops.append(base - roc_auc_score(yte, mdl.predict_proba(Xp)[:, 1]))
            res[g].append(np.mean(drops))
pi = pd.DataFrame({"auc_drop_mean": {g: np.mean(v) for g, v in res.items()}, "auc_drop_se": {g: np.std(v) / np.sqrt(len(v)) for g, v in res.items()}}).sort_values("auc_drop_mean", ascending=False)
print("\n== grouped permutation importance of the tuned elastic net (AUC lost when the group is shuffled, held-out folds)"); print(pi.round(4).to_string())
pi.to_csv("results/insights_permutation_importance.csv")

# ---------------------------------------------------------------- 4. Interactions
print("\n== INTERACTIONS")
# 4a. stratified quit rates
def tab(a, b, la, lb):
    t = pd.crosstab(a, b, values=y, aggfunc="mean"); n = pd.crosstab(a, b)
    t.index.name, t.columns.name = la, lb
    return t.round(3), n
level = X.JobLevel.clip(upper=3).map({1: "L1", 2: "L2", 3: "L3+"})
ten = pd.cut(X.YearsAtCompany, [-1, 1, 3, 10, 99], labels=["<=1y", "2-3y", "4-10y", ">10y"])
trav = np.select([X["BusinessTravel_Travel_Frequently"] == 1, X["BusinessTravel_Non-Travel"] == 1], ["freq", "none"], "rarely")
role3 = np.select([X["JobRole_Sales Representative"] == 1, X["JobRole_Laboratory Technician"] == 1, X["JobRole_Human Resources"] == 1], ["SalesRep", "LabTech", "HR"], "other")
marital = X["MaritalStatus_Single"].map({1: "single", 0: "not single"})
ot = X.Overtime.map({1: "OT", 0: "no OT"})
strat = {}
for nm, (a, b) in {"Overtime x Marital": (ot, marital), "Overtime x Travel": (ot, pd.Series(trav, index=X.index)),
                   "Overtime x Job level": (ot, level), "Overtime x Tenure": (ot, ten), "Overtime x Role": (ot, pd.Series(role3, index=X.index)),
                   "Tenure x Job level": (ten, level), "Marital x Job level": (marital, level)}.items():
    r, n = tab(a, b, a.name or nm.split(" x ")[0], b.name or nm.split(" x ")[1])
    print(f"\n-- quit rate: {nm}"); print(r.to_string()); print("n:"); print(n.to_string())
    strat[nm] = (r, n)

# 4b. likelihood-ratio test for every pairwise interaction among key drivers (added one at a time to the explanatory model)
key = ["Overtime", "Travel_frequently", "Single", "LogMonthlyIncome", "YearsAtCompany", "TotalWorkingYears", "NumCompaniesWorked",
       "DistanceFromHome_10mi", "MostRecentPerfEval", "Role_SalesRep", "Role_LabTech", "YearsSinceLastPromotion"]
Z = (D - D.mean()) / D.std()
for c in ["Overtime", "Travel_frequently", "Single", "Role_SalesRep", "Role_LabTech"]: Z[c] = D[c]   # keep binaries 0/1
base_m = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)
rows = []
for a, b in itertools.combinations(key, 2):
    if set([a, b]) <= {"Role_SalesRep", "Role_LabTech"}: continue
    Zi = Z.copy(); Zi[f"{a}*{b}"] = Z[a] * Z[b]
    mi = sm.Logit(y, sm.add_constant(Zi)).fit(disp=0)
    lr = 2 * (mi.llf - base_m.llf); p = stats.chi2.sf(lr, 1)
    rows.append(dict(pair=f"{a} x {b}", coef=mi.params[f"{a}*{b}"], odds_ratio_of_interaction=np.exp(mi.params[f"{a}*{b}"]), lr_p=p))
it = pd.DataFrame(rows).sort_values("lr_p")
it["p_bonferroni"] = (it.lr_p * len(it)).clip(upper=1)
rk = it.lr_p.rank(); it["q_BH"] = (it.lr_p * len(it) / rk).clip(upper=1)
print(f"\n-- LR test for {len(it)} pairwise interactions (top 8; Bonferroni threshold p<{0.05/len(it):.4f})"); print(it.head(8).round(4).to_string(index=False))
it.to_csv("results/insights_interaction_tests.csv", index=False)

# 4c. does allowing interactions help out of sample? additive vs interacting boosted trees, same budget
cv5 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
cvr = {}
for depth in (1, 2, 3):
    g = LGBMClassifier(n_estimators=300, learning_rate=0.02, max_depth=depth, num_leaves=2**depth, subsample=0.7, subsample_freq=1,
                       colsample_bytree=0.6, min_child_samples=20, reg_lambda=5, random_state=SEED, verbose=-1, n_jobs=1)
    s = cross_val_score(g, X, y, cv=cv5, scoring="roc_auc", n_jobs=2); cvr[f"boosted trees depth {depth} ({'additive' if depth == 1 else 'interactions up to ' + str(depth) + '-way'})"] = (s.mean(), s.std())
lr_add = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(solver="saga", C=0.1, l1_ratio=0.1, max_iter=20000)), X[lin], y, cv=cv5, scoring="roc_auc", n_jobs=2)
cvr["elastic-net logistic (additive)"] = (lr_add.mean(), lr_add.std())
print("\n-- CV AUC: additive vs interacting models"); print(pd.DataFrame(cvr, index=["cv_auc", "cv_std"]).T.round(4).to_string())
pd.DataFrame(cvr, index=["cv_auc", "cv_std"]).T.to_csv("results/insights_additive_vs_interactions.csv")

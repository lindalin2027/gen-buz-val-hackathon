# Retention recommendations for Next Labs

Evidence base: 1,170 training employees (15.8% quit). "EDA" = raw quit rates with 95% intervals. "Model" = odds ratios (OR) from an interpretable logistic regression (CV AUC 0.81) that controls for the other drivers at the same time, plus a permutation test on the tuned elastic net (AUC lost when a factor is shuffled on held-out data). An OR above 1 means more likely to quit.

## Recommendations and evidence

| # | Recommendation | EDA evidence | Model evidence | Strength |
|---|---|---|---|---|
| 1 | Reduce chronic overtime, starting with new hires and level 1 staff | 29.7% quit with overtime vs 10.2% without (n=337 / 833) | OR 5.0 (3.4 to 7.5). Predicted quit rate 10% to 29% if everyone worked overtime, others fixed. Largest permutation importance (AUC drop 0.074) | Strong |
| 2 | Invest in the first year (onboarding, 30/60/90 check-ins) | 35.4% quit in year 1 vs 12.4% at 4 to 10 years and 6.4% beyond 10 | Employees with 2 years or less: OR 2.3 (1.4 to 3.7) holding role, pay, experience fixed. Tenure group permutation 0.023 | Strong |
| 3 | Pilot retention fixes in hot-spot roles: Sales Representative, HR, Laboratory Technician (and Sales Executive) | Sales Rep 40.3% (n=67), HR 25.6% (n=39), Lab Tech 23.5% (n=200) vs 15.8% overall | vs Research Scientist: Sales Rep OR 3.7 (1.9 to 7.5), HR 3.2 (1.3 to 7.8), Lab Tech 2.2 (1.3 to 3.9), Sales Exec 2.3 (1.2 to 4.6). Role permutation 0.042 (second largest) | Strong, small n for Sales Rep and HR |
| 4 | Cap or rotate frequent travel | 25.0% frequent vs 14.7% rarely vs 6.7% none | Frequent vs rarely OR 1.9 (1.2 to 2.9); none OR 0.4. About +7 points predicted | Moderate to strong |
| 5 | Look at job-hopping history as a signal for who needs early support (not as a hiring filter) | 22.6% with 5 or more prior employers vs 14.0% | OR 1.21 per prior employer (1.12 to 1.30). Permutation 0.038 (third) | Moderate |
| 6 | Review entry-level pay and the level 1 to level 2 path, as a pilot | Lowest eighth of pay 33.1% vs 8.9% at $8.4k+. Level 1 26.0% vs level 2 10.6% | Weaker once role, tenure and experience are held fixed: +20% pay gives OR 0.89 (0.80 to 1.01), about -1 point. Pay is largely a marker of level and experience | Association strong, causal effect uncertain |
| 7 | Flexibility for long commutes | 19.6% at 20+ miles vs 15.1% | OR 1.35 per 10 miles (1.08 to 1.69), about +3 points | Modest |
| 8 | Watch promotion stagnation | Weak in EDA | OR 1.10 per year since last promotion (1.01 to 1.20) | Weak to modest |

Not supported: performance rating (OR 0.99, p=0.90, permutation importance about 0; the raw 21% vs 12% gap disappears once other factors are controlled), education, gender. Single status is a strong predictor (OR 2.6, 25.4% vs 11.4%) but should not drive individual treatment.

## Interactions

- **Overtime compounds with other risk factors in absolute terms.** Quit rates with vs without overtime: level 1 51.6% vs 15.2%; first year 55.0% vs 25.2%; single 49.5% vs 15.8%; frequent travel 38.7% vs 17.9%; Sales Rep 66.7% vs 30.6%.
- **Statistically, most of this is multiplicative, not a special synergy.** No pairwise interaction involving overtime survives multiple-testing correction for marital status, travel, role or level. The odds ratio for overtime is similar across those groups, so the extra danger in already high-risk groups comes from stacking.
- **One suggestive interaction: overtime x tenure** (interaction OR 0.52, p=0.005, false discovery q=0.09, not conclusive). The overtime odds ratio is about 3.6 in the first year and about 1.3 after 10 years, so overtime matters most for newer staff.
- **Significant after correction, but not a story:** years at company x total working years, income x years at company, income x total working years. These variables are strongly correlated, so these terms mostly capture curved tenure and pay effects.
- **Interactions barely help prediction.** CV AUC: boosted trees with no interactions 0.790, with 2-way 0.795, with 3-way 0.794 (fold-to-fold SD about 0.03); additive elastic net logistic 0.818. Explicit interaction terms in the logistic gave no gain either (0.817 vs 0.818).
- **Practical implication:** because risks multiply, the same intervention saves the most people in groups that already have several risk factors. Target by combined risk, for example overtime among level 1 staff or first-year employees.

## Caveats

- These are associations, not proven causes. Pilot with a comparison group before rolling out.
- Interpretation of single coefficients is limited by overlap among pay, level and tenure (largest VIF 6.1).
- Several groups are small (Sales Rep 67, HR 39, Managers 90, Research Directors 63). Their intervals are wide.
- Years-at-company and similar fields for people who quit are presumably measured at departure.
- Model ranks well (AUC about 0.77 to 0.82) but probabilities are not calibrated. Use it to prioritize groups, not to make decisions about individuals.

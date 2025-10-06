
# Shell.ai 2025 — Fuel Blend Properties Prediction

**Multi-output regression to predict 10 final blend properties from 55 inputs (5 blend fractions + 50 component COA properties).**  
Built for the **Shell.ai Hackathon for Sustainable & Affordable Energy 2025**.

---

## Problem
Given fuel blends composed of 5 base components and their batch properties (COA), predict the final blend properties:
- Inputs: **55 features** → `5` blend composition columns + `50` component property columns (`Component{1..5}_Property{1..10}`)
- Targets: **10 outputs** → `BlendProperty1 ... BlendProperty10`
- Metric: **Mean Absolute Percentage Error (MAPE)** (lower is better)
- Submission: **500 × 10 CSV** with columns `BlendProperty1,...,BlendProperty10` and no ID

---
## Approach 
1. **EDA**: shape, types, ranges, correlations, target zero checks (important for MAPE).
2. **Preprocessing**:  
   - **Scaling** with `StandardScaler`  
   - **PCA** on the 50 component features to retain ~95% variance  
   - Keep scaled blend fractions + PCA components as the final feature set
3. **Modeling**: multi-output regression with cross-validation (MAPE):  
   - **Ridge** (fast baseline)  
   - **LightGBM** wrapped in `MultiOutputRegressor` (strong baseline)  
4. **Model selection**: choose model with the best mean CV MAPE.
5. **Training on full data** and **predicting** for `test.csv`.
6. **Submission**: write `submission.csv` with exact target columns.

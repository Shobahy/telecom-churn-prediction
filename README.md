# Telecom Customer Churn Prediction

Predicts which customers of a telecom company are likely to cancel their service (churn), so the company can act before they leave.

Final assignment for the **GCI World Data Science Program** (Matsuo-Iwasawa Lab, University of Tokyo).

## Data

- Two tables, **Client** and **Record**, each with 100,000 customers, merged 1:1 on `Customer_ID` into about 100 features
- Features cover billing and revenue, minutes of use and changes in usage, dropped and blocked calls, tenure, handset price and equipment age, and demographics
- The target is balanced: roughly 50% of customers churned

The dataset was provided by the course and is not included in this repository.

## Exploratory analysis

- **Tenure:** churn rate peaks around **11–13 months** with the company, which suggests customers leave when their first-year contract ends
- **Usage:** charts of average usage by tenure and of changes in minutes and revenue between churned and retained customers
- **Equipment age:** compared how long churned and retained customers had owned their handsets

## Approach

- Dropped columns with more than 30% missing values; filled missing categorical values with `"Unknown"`
- Removed the `ethnic` column (see *Fairness check* below)
- Stratified split: 64% train, 16% validation, 20% test
- **CatBoost** classifier (handles categorical features natively), with early stopping on validation AUC
- **SHAP** values to explain which features drive each prediction

## Results

On the 20,000-customer test set:

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.694 |
| Accuracy | 0.64 |
| Precision / Recall (churn) | 0.63 / 0.65 |

The most important predictors were **tenure (months)**, **equipment age**, **change in minutes of use**, monthly recurring charge and average minutes of use.

Churn is notoriously hard to predict from billing and usage data alone. The model performs clearly above the 50% baseline, but there's room for improvement (see *Next steps*).

## Fairness check

The original dataset includes an `ethnic` column, and an earlier version of the model ranked it among its top features. Since a model used to make decisions about real customers should not rely on ethnicity, it was removed from the final model.

| Version | Test ROC-AUC |
|---------|--------------|
| With `ethnic` | 0.695 |
| Without `ethnic` (final) | 0.694 |

The column added essentially nothing to performance, and the main predictors (tenure, equipment age, change in usage) stayed the same.

## Next steps

- Feature engineering around the 12-month contract-renewal window
- Comparison with other models (XGBoost, logistic regression baseline)

## Tech stack

Python, pandas, NumPy, scikit-learn, CatBoost, SHAP, Matplotlib, Seaborn

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Put `Client.csv` and `Record.csv` in a `telecom/` folder next to the notebook
3. Run `telecom_churn_prediction.ipynb`. In Google Colab, create the `telecom/` folder in the file panel and upload the CSVs there. Training takes about 15 minutes.

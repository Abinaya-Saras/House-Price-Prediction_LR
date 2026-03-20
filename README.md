# 🏠 House Price Prediction

A machine learning project to predict house sale prices using advanced regression techniques, feature engineering, hyperparameter tuning, and model explainability.

---

## 📁 Dataset

- **Source:** Kaggle House Prices Competition (`train.csv`, `test.csv`)
- **Target:** `SalePrice` (log-transformed during training)
- **Features:** 80+ features covering size, quality, location, and condition

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)

- Python, Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM
- Matplotlib, Seaborn
- SHAP, LIME (model explainability)
- Optuna (hyperparameter tuning)

---

## 🔍 Project Workflow

1. **Load Data** — Train, test, and sample submission CSVs
2. **EDA** — Distribution of `SalePrice`, missing value analysis
3. **Preprocessing**
   - Fill categorical NAs with `"None"` (e.g., no garage/pool)
   - Fill numeric NAs with `0`
   - Log-transform target: `y = log1p(SalePrice)`
4. **Feature Engineering**
   - `TotalSF` = Basement + 1st Floor + 2nd Floor area
   - `TotalBath` = Full + Half bathrooms (weighted)
   - `HouseAge` and `RemodAge`
   - Binary flags: `HasPool`, `HasGarage`, `HasBsmt`
   - Ordinal encoding for quality columns (e.g., `Ex=5`, `Gd=4`)
   - One-hot encoding via `pd.get_dummies()`
5. **Modeling**
   - Baseline: Lasso, Random Forest, Gradient Boosting
   - Stacking: RF + GBR → Lasso meta-learner
   - Tuned: XGBoost with `RandomizedSearchCV`
6. **Evaluation** — 5-Fold Cross-Validation (RMSE on log-target)
7. **Explainability** — SHAP feature importance + LIME instance explanations
8. **Submission** — Predictions saved to `sample_submission.csv`

---

## 📊 Models & Evaluation

| Model                        | CV RMSE (log) |
|------------------------------|:-------------:|
| Lasso                        | ~0.115        |
| Random Forest                | ~0.140        |
| Gradient Boosting            | ~0.118        |
| **Stacking (RF + GBR)**      | **Best**      |
| XGBoost (Tuned)              | Competitive   |

> Metric: Root Mean Squared Log Error (RMSLE) via 5-Fold CV

---

## 📂 Project Structure
```
├── train.csv
├── test.csv
├── sample_submission.csv
├── House_Price_Prediction_Mlt_Mini_Project.ipynb
├── final_model.pkl
├── shap_summary.png
└── README.md
```

---

## 🚀 How to Run
```bash
git clone https://github.com/Abinaya-Saras/House-Price-Prediction_LR.git
cd house-price-prediction
pip install scikit-learn xgboost lightgbm shap lime optuna matplotlib seaborn
jupyter notebook House_Price_Prediction_Mlt_Mini_Project.ipynb
```

---

## 📌 Key Findings

- **Log-transforming** `SalePrice` improves model performance significantly
- **TotalSF** and **Overall Quality** are the strongest price predictors
- **Stacking** outperforms individual models
- **SHAP** confirms that size, quality, and neighborhood drive prices most

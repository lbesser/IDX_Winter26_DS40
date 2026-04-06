# California Residential Home Price Prediction

A machine learning project that predicts the **close price of single-family residential homes** in California using MLS sold listing data. Built as part of a 12-week internship.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Preprocessing](#preprocessing)
- [Models Tested](#models-tested)
- [Results](#results)
- [Streamlit App](#streamlit-app)

---

## Project Overview

**Goal:** Predict `ClosePrice` for single-family residential properties in California using structured MLS data.

**Property filter applied:**
- `PropertyType` = Residential
- `PropertySubType` = SingleFamilyResidence

**Target variable:** `ClosePrice`

---

## Dataset

- **Source:** CRMLS (California Regional MLS) — monthly `CRMLSSold` CSV files downloaded via FTP from `/raw/California`
- **Date range:** March 2025 – December 2025 (10 months)
- **Files loaded:** `CRMLSSold202503.csv` through `CRMLSSold202512.csv`
- **Raw records after combining:** ~128,905 rows
- **Records after cleaning:** 128,720 rows

**Key columns used:**

| Column | Description |
|---|---|
| `ClosePrice` | Final sale price (target variable) |
| `LivingArea` | Interior square footage |
| `BedroomsTotal` | Number of bedrooms |
| `BathroomsTotalInteger` | Number of bathrooms |
| `LotSizeSquareFeet` | Total lot size in sq ft |
| `YearBuilt` | Year the property was built |
| `ListPrice` | Original listing price |
| `DaysOnMarket` | How long the listing was active |
| `ParkingTotal` | Number of parking spaces |
| `CloseDate` | Date the sale closed |

---

## Repository Structure

```
california-price-prediction/
│
├── notebook02_preprocessing.ipynb     # Data cleaning, encoding, train/test split
├── 03_baseline_model.ipynb            # Linear Regression baseline
├── 04_additional_model.ipynb          # Decision Tree model
├── 05_advanced_models.ipynb           # XGBoost & LightGBM with tuning
├── 06_evaluation.ipynb                # Full evaluation: MAPE, MdAPE, price bands
│
├── property_data_train_x.csv          # Cleaned training features
├── property_data_test_x.csv           # Cleaned test features
├── property_data_train_y.csv          # Training labels (ClosePrice)
├── property_data_test_y.csv           # Test labels (ClosePrice)
│
├── app.py                             # Streamlit prediction app (Week 9)
└── README.md
```

---

## Setup & Installation

**Requirements:** Python 3.9+

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn category_encoders streamlit joblib
```

If running notebooks in Google Colab, mount your Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## How to Run

### Run the notebooks in order:

```bash
# 1. Preprocessing
jupyter notebook notebook02_preprocessing.ipynb

# 2. Baseline model
jupyter notebook 03_baseline_model.ipynb

# 3. Decision Tree
jupyter notebook 04_additional_model.ipynb

# 4. Advanced models (XGBoost / LightGBM)
jupyter notebook 05_advanced_models.ipynb

# 5. Evaluation
jupyter notebook 06_evaluation.ipynb
```

### Launch the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

---

## Preprocessing

Done in `notebook02_preprocessing.ipynb`:

1. **Loaded** 10 monthly CSV files and concatenated into a single DataFrame
2. **Filtered** to `PropertyType = Residential` and `PropertySubType = SingleFamilyResidence`
3. **Removed invalid rows** — dropped nulls in `ClosePrice`, duplicates, and logic errors (e.g., living area < 100 sqft, price ≤ 0), removing 185 rows
4. **Dropped sparse columns** — columns with near-100% missing values dropped: `FireplacesTotal`, `AboveGradeFinishedArea`, `TaxAnnualAmount`, `TaxYear`, `BelowGradeFinishedArea`, `BuilderName`
5. **Encoded categoricals** — high-cardinality columns (`City`, `PostalCode`, `MLSAreaMajor`, agent/office names) encoded using Binary Encoding; boolean fields mapped to 0/1
6. **Imputed** remaining numeric NaNs using mean imputation
7. **Scaled** numeric features using `StandardScaler` (fit on train only, applied to test)
8. **Train/test split** — most recent month (December 2025) held out as the test set
   - Training rows: **118,391**
   - Test rows: **10,329**

---

## Models Tested

| Notebook | Model | Notes |
|---|---|---|
| `03_baseline_model.ipynb` | Linear Regression | Baseline; scaled numeric features only |
| `04_additional_model.ipynb` | Decision Tree | No scaling needed; imputed features |
| `05_advanced_models.ipynb` | XGBoost | Grid search over depth, learning rate, n_estimators |
| `05_advanced_models.ipynb` | LightGBM | Same grid search; faster training |

**Features used across all models:**
`ListPrice`, `DaysOnMarket`, `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, `ParkingTotal`, `LotSizeSquareFeet`, `YearBuilt`

---

## Results

| Model | R² | MAE | MAPE | MdAPE |
|---|---|---|---|---|
| Linear Regression | 0.976 | $53,032 | — | — |
| Decision Tree | 0.948 | $75,165 | — | — |
| **XGBoost** | **0.886** | — | **13.45%** | **9.34%** |
| LightGBM | 0.882 | — | 13.89% | 9.65% |

> Note: Linear Regression's high R² reflects that `ListPrice` (a near-perfect proxy for `ClosePrice`) was included as a feature. XGBoost and LightGBM were evaluated without it for a more realistic test.

**XGBoost MAPE by price band:**

| Price Band | Range | MAPE |
|---|---|---|
| Low | < $200K | 94.96% |
| Mid | $200K – $500K | 18.07% |
| High | $500K – $1M | 11.30% |
| Luxury | > $1M | 13.99% |

The model performs best on mid-to-high price homes. Low-priced properties have high error, likely due to data sparsity in that range for California.

**Best overall model: XGBoost** (R² = 0.886, MdAPE = 9.34%)

---

## Streamlit App

`app.py` provides a simple UI where users can input property details and get a predicted close price.

**Inputs:** Living Area, Bedrooms, Bathrooms, Lot Size, Year Built, Days on Market, List Price, Parking

**Output:** Predicted Close Price

The app loads the trained XGBoost model using `joblib`. Make sure `model.pkl` (or equivalent saved model file) is in the same directory as `app.py` before launching.

---

*Project completed as part of a 12-week data science internship — 2025.*

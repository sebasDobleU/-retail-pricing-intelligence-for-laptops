# Laptop Price Prediction — ML Regression Model

> **Retail pricing intelligence**: Given a laptop's technical specs, predict its fair market price within ~20% accuracy using an ensemble of 375 trained models.

---

## Results

| Metric | Value |
|--------|-------|
| Best Model | Gradient Boosting Regressor |
| MAPE (test) | 19.6% |
| MAE (test) | $247 (95% CI: $207–$266) |
| Dataset size | 2,160 laptops |
| Models compared | 375 (Decision Tree + Random Forest + Gradient Boosting) |
| Statistical validation | Wilcoxon test vs. baseline (p ≈ 0) |

> For laptops averaging **$1,300**, a $247 mean error is commercially viable for competitive pricing decisions.

---

## Problem

A laptop retailer needed a data-driven way to set competitive prices for its inventory — both new and refurbished units. Manually pricing each laptop based on spec sheets was slow and inconsistent.

**Goal:** Build a regression model that estimates a laptop's sale price from its technical characteristics, so the sales team can price inventory instantly and accurately.

---

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, scikit-learn, SHAP, matplotlib, seaborn
- **Models:** Decision Tree, Random Forest, Gradient Boosting
- **Deployment:** Desktop GUI (tkinter) for real-time price prediction

---

## Dataset

- **Source:** Public laptop pricing dataset
- **Size:** 2,160 rows × 12 columns
- **Target variable:** `Final Price` (USD)

| Feature | Type | Description |
|---------|------|-------------|
| Brand | Categorical | Manufacturer (Asus, Dell, HP, Lenovo, etc.) |
| Model | Categorical | Specific product line (top 10 + "Other") |
| CPU | Categorical | Processor family (Intel Core i5/i7/i9, AMD Ryzen) |
| RAM | Numeric (GB) | Memory capacity |
| Storage | Numeric (GB) | Disk capacity |
| Storage Type | Categorical | SSD, HDD, eMMC |
| GPU | Categorical | Dedicated GPU model or integrated |
| Screen | Numeric (in) | Display size in inches |
| Touch | Binary | Touchscreen yes/no |
| Status | Binary | New or refurbished |

---

## Methodology

### 1. Data Cleaning & Quality Analysis
- Identified and handled missing values (GPU: 63% missing → valid "no dedicated GPU"; Storage type: 2% missing)
- Removed outliers: screen size = 0 in, storage = 0 GB, prices > $6,000
- Validated business consistency rules (e.g., low-end CPU should not pair with dedicated GPU)

### 2. Feature Engineering
- Applied **One-Hot Encoding** to Brand, Storage Type, CPU, GPU
- Applied **Ordinal Encoding** to Touch and Status
- Reduced Model column from 121 categories to top 10 + "Other"
- Final feature matrix: **2,154 rows × 120 columns**

### 3. Correlation & Feature Importance (SHAP)
- Computed Pearson and Spearman correlations
- Used **SHAP values** for model-agnostic feature importance
- Key finding: **RAM and Storage** are the dominant price drivers

![SHAP bar chart showing RAM and Storage as top features]

### 4. Model Selection — 375 Models Compared
Trained and evaluated across 6 feature subsets:
- **125 Decision Tree** models (varying depth, min_samples_leaf)
- **125 Random Forest** models
- **125 Gradient Boosting** models

Selected the best by lowest **MAPE** on validation set.

**Winning configuration:**
```
GradientBoostingRegressor(
    max_depth=5,
    min_samples_leaf=2,
    n_estimators=300,
    random_state=40
)
```

### 5. Evaluation
- Train / Validation / Test split: **65% / 20% / 15%**
- No significant overfitting: val MAPE (19.48%) ≈ test MAPE (19.61%)
- Wilcoxon signed-rank test confirms model significantly outperforms mean-price baseline (p ≈ 3.28 × 10⁻⁴⁹)

---

## Key Insights

- **RAM** is the single most influential feature — more RAM consistently means higher price
- **Storage capacity** is the second strongest driver
- **Screen size** has a non-linear effect: very large screens don't always mean higher price
- **Refurbished** status reliably lowers predicted price vs. new equivalents
- **Asus** laptops had the lowest average prediction error (~$200), while Samsung had the highest (~$800)

---

## Deployment

A desktop GUI lets users input specs and receive an instant price estimate:

```
Status:           New
Brand:            MSI
Model:            Titan
CPU:              AMD Radeon
RAM (GB):         32
Storage (GB):     2000
Storage Type:     HDD
GPU:              No GPU
Screen (in):      15.6
Touchscreen:      No

→ Estimated Price: $1,851.26
```

---

## Project Structure

```
laptop-price-prediction/
├── data/
│   └── laptops.csv
├── notebooks/
│   └── 01_eda.ipynb
│   └── 02_feature_engineering.ipynb
│   └── 03_model_selection.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   └── train.py
│   └── predict.py
│   └── gui.py
├── models/
│   └── best_model.pkl
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/your-username/laptop-price-prediction
cd laptop-price-prediction
pip install -r requirements.txt

# Run the GUI
python src/gui.py

# Or run predictions from CLI
python src/predict.py --brand Asus --ram 16 --storage 512 --cpu "Intel Core i7"
```

---

## Contact

Built by [Your Name] — open to freelance data science and ML projects.  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

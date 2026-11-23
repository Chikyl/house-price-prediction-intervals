# House Price Prediction Intervals: CatBoost + LightGBM + XGBoost Ensemble

A gradient-boosting ensemble that predicts **90% prediction intervals** (a lower and an upper bound) for house sale prices, instead of a single point estimate. Three GPU-accelerated frameworks are trained, validated with 5-fold cross-validation, and blended with a hill-climbing weight search that minimizes the **Mean Winkler Interval Score (MWIS)**.

Built for the Kaggle *Prediction Interval Competition II: House Price*.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Evaluation Metric: Winkler Interval Score](#evaluation-metric-winkler-interval-score)
- [Ensemble Optimization (Hill Climbing)](#ensemble-optimization-hill-climbing)

---

## Overview

A point prediction says "this house will sell for $800k". A prediction interval says "we are 90% confident it will sell between $650k and $950k". The second is far more useful for risk-aware decisions such as lending, valuation and investing.

**What this project does**

| Step | Description |
|------|-------------|
| 1 | Loads 200,000 property sales and drops columns with many missing values |
| 2 | Engineers features: property age, renovation status, total living area, total bathrooms, total assessed value |
| 3 | Trains **LightGBM**, **CatBoost** and **XGBoost** to produce lower and upper bounds |
| 4 | Validates with **5-fold CV**, scoring each model by MWIS and empirical coverage |
| 5 | Searches ensemble weights by **hill climbing** to minimize MWIS |
| 6 | Retrains on all the data and writes `submission.csv` with `pi_lower` and `pi_upper` |

---

## Results

All numbers are **out-of-fold (OOF)** results from 5-fold cross-validation on the 200,000-row training set (target interval coverage: 90%, α = 0.10). **Lower MWIS is better.**

### Individual models

| Model | Interval method | MWIS ↓ | Coverage |
|-------|-----------------|-------:|---------:|
| LightGBM | Two quantile regressors (5th and 95th percentile) | **340,783.69** | 83.50% |
| XGBoost | Two quantile regressors (5th and 95th percentile) | 348,202.01 | 85.68% |
| CatBoost | `RMSEWithUncertainty`, mean ± 1.645σ | 352,068.85 | 88.06% |

### Ensemble

| Model | MWIS ↓ |
|-------|-------:|
| Best single model (LightGBM) | 340,783.69 |
| Initial blend (0.4 / 0.3 / 0.3) | 330,347.13 |
| **Optimized ensemble (hill climbing)** | **329,338.31** |

- The optimized ensemble improves on the best single model by about **11,445 MWIS points (≈ 3.4%)**.
- It also improves on the hand-picked initial weights by about **1,009 points (≈ 0.3%)**.

### Optimized ensemble weights

| Model | Weight |
|-------|-------:|
| LightGBM | 0.4623 |
| CatBoost | 0.3754 |
| XGBoost | 0.1623 |

### Hill-climbing trajectory

The search ran for 100 steps and accepted 24 improvements, with the score falling from 330,376.78 at step 0 to 329,338.31 at step 97:

| Step | Best MWIS | LightGBM | CatBoost | XGBoost |
|-----:|----------:|---------:|---------:|--------:|
| 0 | 330,376.78 | 0.418 | 0.289 | 0.293 |
| 15 | 329,790.01 | 0.361 | 0.391 | 0.248 |
| 49 | 329,451.39 | 0.424 | 0.383 | 0.193 |
| 80 | 329,393.00 | 0.444 | 0.377 | 0.179 |
| 97 | **329,338.31** | **0.462** | **0.375** | **0.162** |

### Key takeaways

- **No single model wins on every axis.** LightGBM has the best MWIS, but CatBoost's Gaussian intervals have the best coverage. Blending the two trades their strengths off against each other.
- **Individual quantile models under-cover.** LightGBM (83.5%) and XGBoost (85.7%) fall below the 90% target. Coverage for the final blended interval was not computed in the notebook.
- **The optimizer moves weight toward LightGBM and CatBoost** and away from XGBoost, the opposite of the starting prior.

---

## Pipeline

1. **Raw CSVs**: 200k training rows
2. **Drop sparse columns**: `submarket`, `subdivision`, `sale_nbr`
3. **Feature engineering**: age, renovation, living area, baths, total value
4. **Ordinal encoding** of categorical columns
5. **Train three models**: LightGBM (q=0.05 / 0.95), CatBoost (mean ± 1.645σ), XGBoost (q=0.05 / 0.95)
6. **5-fold out-of-fold predictions**
7. **Hill-climbing weight search** on MWIS
8. **Retrain on full data** and blend with the optimized weights
9. **Sanity checks** on the intervals, then write `submission.csv`

---

## Dataset

Source: Kaggle competition *Prediction Interval Competition II: House Price*, residential property sales (King County, WA area, based on the coordinates and cities in the data).

| File | Purpose |
|------|---------|
| `dataset.csv` | Training data: 200,000 rows × 47 columns, including the target `sale_price` |
| `test.csv` | Test data: same features without the target |

Representative columns: `sale_date`, `sale_price`, `sale_warning`, `join_status`, `join_year`, `latitude`, `longitude`, `area`, `city`, `zoning`, `sqft`, `sqft_fbsmt`, `bath_full`, `bath_3qtr`, `bath_half`, `land_val`, `imp_val`, `year_built`, `year_reno`, and view indicators such as `view_rainier`, `view_olympics` and `view_cascades`.

**Columns dropped:** `submarket`, `subdivision` and `sale_nbr` (a large share of missing values), plus `id` and the raw `sale_date` after feature extraction. The models use **49 features** after engineering.

> The data is not redistributed in this repo. Download it from the Kaggle competition page.

---

## Feature Engineering

Implemented in the `preprocess()` function:

| Feature | Definition |
|---------|-----------|
| `sale_year`, `sale_month` | Extracted from `sale_date` |
| `age` | `sale_year − year_built` |
| `renovated` | `1` if `year_reno > 0`, otherwise `0` |
| `years_since_reno` | `sale_year − year_reno` for renovated homes, otherwise `0` |
| `total_baths` | `bath_full + 0.75 × bath_3qtr + 0.5 × bath_half` |
| `total_value` | `land_val + imp_val` |
| `living_area` | `sqft + sqft_fbsmt` |

**Categorical handling:** `sale_warning`, `join_status`, `city` and `zoning` are ordinal-encoded with `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`. It is fit on the training data and applied to the test data, so unseen categories at test time are safe.

---

## Modeling Approach

Each framework produces a 90% interval in the way it handles best:

### LightGBM and XGBoost: direct quantile regression
Two separate regressors per framework: one for the **5th percentile** (lower bound) and one for the **95th percentile** (upper bound).

| Hyperparameter | LightGBM | XGBoost |
|----------------|:--------:|:-------:|
| Objective | `quantile` | `reg:quantileerror` |
| Quantiles | 0.05 / 0.95 | 0.05 / 0.95 |
| `n_estimators` | 1500 | 1500 |
| `learning_rate` | 0.05 | 0.05 |
| Tree size | `num_leaves=63` | `max_depth=8` |
| `subsample` | 0.8 (`subsample_freq=1`) | 0.8 |
| Device | GPU | GPU (`gpu_hist`) |
| `random_state` | 42 | 42 |

### CatBoost: uncertainty-aware regression
A single model trained with `loss_function="RMSEWithUncertainty"` predicts both a **mean** and a **variance**. Assuming normally distributed errors, the interval is:

```
lower = mean − 1.645 × σ
upper = mean + 1.645 × σ
```

where 1.645 is the z-score for a 90% two-sided interval. Settings: 1500 iterations, `learning_rate=0.05`, `depth=8`, GPU, `random_seed=42`.

### Post-processing
- Bounds are re-ordered with `min`/`max`, so the lower bound never exceeds the upper bound.
- Final lower bounds are clipped at `0`, since prices cannot be negative.

---

## Evaluation Metric: Winkler Interval Score

The **Winkler score** rewards narrow intervals and penalizes misses in proportion to how far the true value falls outside:

```
WIS = (U − L) + (2/α) · max(L − y, 0) + (2/α) · max(y − U, 0)
```

With α = 0.10, every unit of miss is penalized by a factor of **20**. The **Mean Winkler Interval Score (MWIS)** averages this over all samples. It balances two competing goals:

- **Sharpness**: tighter intervals are better.
- **Calibration**: intervals must actually cover the true price.

Empirical **coverage** (the share of true prices inside the interval) is tracked alongside it.

---

## Ensemble Optimization (Hill Climbing)

Lower bounds and upper bounds are blended with the **same weight vector**:

```
final_lower = Σ wᵢ · lowerᵢ        final_upper = Σ wᵢ · upperᵢ
```

**Algorithm** (100 steps, starting from `[0.4, 0.3, 0.3]`):

1. Perturb the current weights with a scaled Dirichlet-based random step.
2. Clip negatives to zero and renormalize so the weights sum to 1.
3. Compute the MWIS of the blended OOF

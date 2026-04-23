# F1 Podium Predictor — End-to-End ML Pipeline

## Business Problem
Given pre-race information available before lights out, predict the 
probability of each driver finishing on the podium (Top 3).

Applicable to race strategy simulation, fantasy sports optimization, 
and probabilistic outcome modeling under uncertainty — the same 
framework used in financial risk scenario analysis.

## Data
- **Source:** FastF1 API (official F1 timing data)
- **Period:** 2018–2021 (4 seasons, 75 races, ~1,500 driver-race observations)
- **Target:** Binary — podium finish (Top 3) vs. not (~15% positive rate)

## Project Structure
f1-podium-predictor/
├── data/
│   ├── raw/                  # Raw race + qualifying data from FastF1
│   └── processed/            # Train/val splits
├── notebooks/
│   ├── 01_data_collection    # FastF1 API pipeline
│   ├── 02_eda                # Exploratory data analysis (5 plots)
│   ├── 03_feature_engineering# Rolling features + leakage prevention
│   ├── 04_modeling           # 3-model comparison + evaluation
│   └── 05_explainability     # SHAP global + local explanations
├── outputs/                  # All saved plots and best model
└── requirements.txt

## Methodology

### Feature Engineering
| Feature | Type | Description |
|---|---|---|
| GridPosition | Pre-race | Starting grid position |
| QualiGapToPole_pct | Pre-race | % slower than pole in qualifying |
| IsWet | Pre-race | Rain during race (binary) |
| SafetyCarDeployed | Contextual | Safety car deployed (binary) |
| AvgFinish_Last3 | Rolling | Driver avg finish position, last 3 races |
| DNF_Rate_Last5 | Rolling | Driver DNF rate, last 5 races |
| PodiumRate_Last5 | Rolling | Driver podium rate, last 5 races |
| Constructor_AvgPts_Last3 | Rolling | Constructor avg points, last 3 races |
| SeasonProgress | Contextual | Race number / total races (0–1) |

### Leakage Prevention
All rolling features computed with `shift(1)` — race N only 
sees data from races N-1 and earlier. Train/val split is 
**temporal** (2018–2020 train / 2021 val), never random, 
mirroring real-world prediction constraints.

### Modeling
- **Preprocessor:** Scikit-learn Pipeline (median imputation → standard scaling)
- **Models compared:** Logistic Regression (baseline), XGBoost, LightGBM
- **Class imbalance:** `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (LightGBM)
- **Evaluation:** AUC-ROC + PR-AUC (PR-AUC reported because AUC-ROC is 
  optimistic on imbalanced datasets)

## Results

| Model | AUC-ROC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.8853 | 0.5944 |
| XGBoost | **0.9041** | **0.6584** |
| LightGBM | 0.8787 | 0.5952 |

**Winner: XGBoost** — AUC-ROC 0.9041 on unseen 2021 season data.

Confusion matrix (2021 validation):
- Recall: 70.4% — caught 38 of 54 actual podiums
- Precision: 55.9% — when we predicted podium, correct 56% of the time

## Explainability — SHAP
Used SHAP TreeExplainer to explain all predictions globally and locally.

**Key finding:** GridPosition is the dominant feature but rolling 
form features (AvgFinish_Last3, PodiumRate_Last5) add significant 
predictive signal beyond grid position alone. Wet race flag captures 
chaotic outcomes that static features miss.

Waterfall plots show exactly why the model predicted a podium for 
a specific driver who started outside the top 5 — translatable to 
any stakeholder without ML background.

## Reproduction
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_collection.ipynb
```

Run notebooks in order: 01 → 02 → 03 → 04 → 05

## Key Interview Talking Points
- **Why temporal split?** Random split leaks future race results into 
  training — the model would train on 2021 data and validate on 2020, 
  which is impossible in production.
- **Why PR-AUC alongside AUC-ROC?** With 15% positive class, a naive 
  model scores ~0.85 AUC-ROC by always predicting negative. PR-AUC 
  penalises this and gives a more honest picture.
- **Why SHAP?** OW presents model outputs to C-suite clients. A black-box 
  XGBoost score means nothing to a CEO — a waterfall plot explaining 
  "this driver's strong recent form offset his poor grid position" does.
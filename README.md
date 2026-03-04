
###  Pre-Screening-System-for-Fertility-Assessment-in-Reproductive-Age-Women
This project presents a Machine Learning-based pre-screening system designed to predict early-stage fertility chances in women of reproductive age. The model integrates biochemical biomarkers, clinical history, and lifestyle factors to provide a data-driven fertility risk assessment .

 
#### A Machine Learning Pre-Screening Approach Integrating Biochemical Biomarkers, Clinical History, and Lifestyle Predictors

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-Explainable_AI-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SMOTE-Class_Balancing-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Research_Paper_In_Preparation-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Target-🟢 High / 🟡 Moderate / 🔴 Low Chance-informational?style=flat-square"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Input-Real Patient CSV Dataset-informational?style=flat-square"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Features-25 Original + 7 Engineered-informational?style=flat-square"/>
</p>

---

## Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Feature Categories](#-feature-categories)
- [Pipeline Overview](#-pipeline-overview)
- [Models Used](#-models-used)
- [Evaluation Strategy](#-evaluation-strategy)
- [Explainability — SHAP](#-explainability--shap)
- [Patient Prediction Card](#-patient-prediction-card)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Limitations and Future Work](#-limitations-and-future-work)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## About the Project

This Project is an end-to-end machine learning pipeline for **early-stage, pre-clinical fertility chance prediction** in women of reproductive age.

The tool reads a real patient CSV dataset (`fertility_dataset.csv`), processes 25 clinical, hormonal, and lifestyle features, and predicts each patient's **fertility chance** as one of three classes:

| Class | Meaning |
|---|---|
| 🟢 **High Chance** | Strong positive fertility indicators across most domains |
| 🟡 **Moderate Chance** | Some concerns present — monitoring or mild intervention may help |
| 🔴 **Low Chance** | Multiple factors affecting fertility — specialist consultation recommended |

> **Clinical Gap Addressed:** Most women do not seek fertility advice until after prolonged unsuccessful conception. This tool is designed to flag concerns **before a clinic visit**, enabling earlier referral and intervention.

### Key Contributions
- **Positive framing** — predicts *fertility chance* not *infertility risk* — more empowering for patients
- **Pre-screening context** — designed for use before clinical consultation, not as a diagnostic tool
-  **Real CSV input** — reads directly from `fertility_dataset.csv`
- **Median imputation with missing indicators** — `SimpleImputer(strategy="median", add_indicator=True)` handles skewed biomarker distributions and preserves missingness information
- **Leakage-free pipeline** — imputer and scaler fit on training data only
-  **SMOTE on train only** — class balancing without contaminating the test set
- **SHAP explainability** — colour-coded feature importance (green = positive, red = negative for fertility)
- **Patient prediction card** — visual probability bar chart with profile summary



## Dataset

| Property | Detail |
|---|---|
| **File** | `fertility_dataset.csv` |
| **Source** | Real patient dataset |
| **Original Features** | 25 |
| **Engineered Features** | 7 (added during preprocessing) |
| **Total Features Used** | 32+ (including missing indicator columns) |
| **Target Column** | `fertility_chance` — High / Moderate / Low |
| **Missing Data** | Lab values: AMH, FSH, TSH, Vit D |
| **Imputation Strategy** | Median (robust to skewed distributions) |

### Why Median Imputation?
AMH and FSH follow **right-skewed lognormal distributions** in real clinical populations. Using median instead of mean prevents extreme values from distorting imputed estimates. Setting `add_indicator=True` in `SimpleImputer` also adds binary flag columns for each imputed feature, preserving the information that a value was missing — which itself can be a clinically meaningful signal.

---

## Feature Categories

###  Biochemical Biomarkers (4 features)
| Feature | Unit | Clinical Significance |
|---|---|---|
| `amh` | ng/mL | Anti-Müllerian Hormone — ovarian reserve |
| `fsh` | IU/L | Follicle Stimulating Hormone Day-3 — pituitary-ovarian axis |
| `tsh` | mIU/L | Thyroid Stimulating Hormone — thyroid-reproductive axis |
| `vitd` | ng/mL | Vitamin D (25-OH) — reproductive outcomes |

> All 4 contain missing values — handled with **`SimpleImputer(strategy="median", add_indicator=True)`**

### Clinical History (15 features)
| Feature | Encoding | Notes |
|---|---|---|
| `female_age` | Continuous | 18–45 years |
| `partner_age` | Continuous | Years |
| `months_trying` | Continuous | Duration of conception attempts |
| `cycle_length` | Continuous | Days |
| `cycle_regular` | bool → int | Cast with `.astype(bool)` |
| `ever_pregnant` | bool → int | Cast with `.astype(bool)` |
| `live_births` | Integer | Count |
| `miscarriages` | Integer | Count |
| `ectopic` | bool → int | Cast with `.astype(bool)` |
| `pcos` | bool → int | Cast with `.astype(bool)` |
| `endometriosis` | bool → int | Cast with `.astype(bool)` |
| `thyroid` | bool → int | Cast with `.astype(bool)` |
| `diabetes` | bool → int | Cast with `.astype(bool)` |
| `semen_analysis` | Ordinal | normal=2, unknown=1, abnormal=0 |
| `prior_infertility_dx` | Ordinal | none=2, one_partner=1, both=0 |

###  Lifestyle Predictors (6 features)
| Feature | Encoding | Notes |
|---|---|---|
| `bmi` | Continuous | From height/weight |
| `smoking` | Ordinal | never=0, former=1, current=2 |
| `alcohol` | Continuous | Drinks per week |
| `activity` | Ordinal | sedentary=0, light=1, moderate=2, high=3 |
| `sleep` | Continuous | Hours per night |
| `stress` | Integer | 1–10 scale |

###  Engineered Features (7 features — added in preprocessing)
| Feature | Condition | Clinical Rationale |
|---|---|---|
| `prime_age` | age 22–32 | Peak reproductive window |
| `healthy_bmi` | BMI 18.5–25 | Optimal weight for fertility |
| `good_amh` | AMH ≥ 1.5 ng/mL | Adequate ovarian reserve |
| `normal_fsh` | FSH ≤ 8.0 IU/L | Normal pituitary function |
| `optimal_vitd` | Vit D ≥ 30 ng/mL | Sufficient vitamin D |
| `healthy_habits` | Sum: non-smoker + alcohol≤3 + sleep≥7 + active | Composite lifestyle score (0–4) |
| `no_diagnosis` | No PCOS, endo, thyroid, or diabetes | Absence of fertility-affecting conditions |

---

##  Pipeline Overview

```
fertility_dataset.csv
         │
         ▼
┌─────────────────────────────────┐
│  Load Data                      │  pd.read_csv()
│  df.head(), df.info()           │
│  df.isnull().sum()              │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Boolean Casting                │  7 columns → .astype(bool)
│                                 │  ever_pregnant, cycle_regular,
│                                 │  ectopic, pcos, endometriosis,
│                                 │  thyroid, diabetes
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  EDA (3 sections)               │  Section 1: 3×4 boxplot grid
│                                 │  (age, AMH, BMI, FSH, TSH,
│                                 │   VitD, sleep, alcohol,
│                                 │   cycle, stress, months_trying)
│                                 │  Section 2: Condition prevalence
│                                 │  Section 3: Correlation heatmap
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  IQR Outlier Capping            │  bmi, alcohol, months_trying
│                                 │  Clip: Q1−3×IQR to Q3+3×IQR
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Ordinal Encoding               │  smoking, activity,
│                                 │  semen_analysis,
│                                 │  prior_infertility_dx
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Target Encoding                │  high=2, moderate=1, low=0
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Feature Engineering            │  7 new features added
│                                 │  (prime_age, healthy_bmi,
│                                 │   good_amh, normal_fsh,
│                                 │   optimal_vitd,
│                                 │   healthy_habits, no_diagnosis)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stratified Train/Test Split    │  80% train / 20% test
│  (BEFORE imputation)            │  random_state=42
│                                 │  Prevents data leakage
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  SimpleImputer                  │  strategy = "median"
│  (fit on train only)            │  add_indicator = True
│                                 │  → adds missingness flag cols
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  StandardScaler                 │  fit_transform on train
│  (fit on train only)            │  transform on test
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  SMOTE                          │  k_neighbors=5
│  (applied to train only)        │  Balances all 3 classes
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  4 ML Classifiers               │  Logistic Regression
│  5-Fold Stratified CV           │  Random Forest (n=200)
│  Macro F1 scoring               │  Gradient Boosting (n=200)
│  Auto-selects best by CV F1     │  XGBoost (n=200)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Evaluation                     │  Confusion matrix (cmap=Greens)
│                                 │  Per-class F1 bar chart
│                                 │  ROC curves (one-vs-rest)
│                                 │  Full classification report
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  SHAP Explainability            │  TreeExplainer
│  (best model, 300 test samples) │  Beeswarm — High Chance class
│                                 │  Bar summary plot
│                                 │  Green = positive for fertility
│                                 │  Red = negative for fertility
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Patient Prediction Card        │  Edit patient dict → re-run
│                                 │  Probability bars (H/M/L)
│                                 │  Profile summary table
│                                 │  Confidence score + model name
└─────────────────────────────────┘
```

---

## Models Used

| Model | Configuration | Type |
|---|---|---|
| **Logistic Regression** | `max_iter=1000, random_state=42` | Linear baseline |
| **Random Forest** | `n_estimators=200, random_state=42` | Ensemble bagging |
| **Gradient Boosting** | `n_estimators=200, random_state=42` | Sequential boosting |
| **XGBoost** | `n_estimators=200, eval_metric='mlogloss', random_state=42` | Optimised gradient boosting |

All four models are trained on SMOTE-balanced training data and evaluated on the same held-out test set. The best model is auto-selected by highest mean CV Macro F1.

---

## Evaluation Strategy

| Metric | Role | Why |
|---|---|---|
| **Macro F1** | Primary | Treats all 3 classes equally — critical for imbalanced medical tasks |
| **Accuracy** | Secondary | Overall correctness across all classes |
| **AUC-ROC** | Per-class | One-vs-rest — measures ranking quality for each class |
| **5-Fold Stratified CV** | Model selection | Each fold preserves class proportions |
| **Confusion Matrix** | Error analysis | Shows which classes are confused with each other |
| **Classification Report** | Full breakdown | Precision, recall, F1 per class |

---

## Explainability — SHAP

`TreeExplainer` is applied to the best model on the first 300 test samples.

**Two plots produced:**

| Plot | What it shows |
|---|---|
| **Beeswarm** | Each dot = one patient. X-axis = SHAP value (impact on High Chance prediction). Colour = feature value (high/low). |
| **Bar summary** | Mean absolute SHAP values — overall feature ranking |

**Feature importance colour coding:**
```
🟢 Green — positive fertility indicators
          amh, good_amh, prime_age, healthy_bmi,
          normal_fsh, cycle_regular, vitd,
          semen_analysis, no_diagnosis

🔴 Red   — negative fertility indicators
          pcos, endometriosis, months_trying,
          stress, smoking, bmi extremes
```

---

##  Patient Prediction Card

The prediction cell generates a two-panel visual card for any patient:

**Left panel — Probability bar chart**
```
🟢 High Chance     ████████████  72.3%  ← winner (bold border)
🟡 Moderate Chance ████          18.1%
🔴 Low Chance      ██             9.6%
```

**Right panel — Patient profile summary**
```
Age          34 yrs   Concern
BMI          32       Concern
AMH          6.1      Ok
FSH          3.2      Ok
Smoking      Yes      Danger Zone
Semen        Normal   Ok
Cycle        Regular  Ok
Stress       3/10     Ok
```

**Footer**
```
Result: Moderate Chance  |  Confidence: 72.3%  |  Model: XGBoost
```

---

## Project Structure

```
fertipredict/
│
├── 📓 Fertility_Chance_Prediction.ipynb  ← Main pipeline notebook
│
├── 📊 fertility_dataset.csv              ← Input dataset (required)
│
│
└── README.md                            ← You are here
```

---

## How to Run

### Option 1 — Google Colab (Recommended)

```
1. Upload fertility_dataset.csv to your Colab session
2. Go to colab.research.google.com
3. File → Upload notebook → Fertility_Chance_Prediction.ipynb
4. Confirm data path in Cell 3:
      df = pd.read_csv("/content/fertility_dataset.csv")
5. Runtime → Run all
```

### Option 2 — Run Locally

```bash
git clone https://github.com/rnehareddy74/fertipredict.git
cd fertipredict
pip install -r requirements.txt
jupyter notebook Fertility_Chance_Prediction.ipynb
```

### Option 3 — Predict for a New Patient

Edit the patient dictionary in the prediction cell:

```python
patient = {
    'female_age'    : 29,    # years
    'partner_age'   : 31,
    'months_trying' : 4,
    'cycle_length'  : 28,
    'cycle_regular' : 1,     # 1=regular, 0=irregular
    'ever_pregnant' : 0,
    'live_births'   : 0,
    'miscarriages'  : 0,
    'ectopic'       : 0,
    'pcos'          : 0,
    'endometriosis' : 0,
    'thyroid'       : 0,
    'diabetes'      : 0,
    'bmi'           : 22.0,
    'smoking'       : 0,     # 0=never, 1=former, 2=current
    'alcohol'       : 1,
    'activity'      : 2,     # 0=sedentary,1=light,2=moderate,3=high
    'sleep'         : 7.5,
    'stress'        : 3,
    'amh'           : 3.5,   # np.nan if not available
    'fsh'           : 5.8,   # np.nan if not available
    'tsh'           : 2.0,   # np.nan if not available
    'vitd'          : 40,    # np.nan if not available
    'semen_analysis': 2,     # 2=normal, 1=unknown, 0=abnormal
    'prior_infertility_dx': 2  # 2=none, 1=one_partner, 0=both
}
```

---

## `requirements.txt`

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11
shap>=0.43
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
jupyter>=1.0
```

---

## Limitations and Future Work

### Current Limitations
```
1.No hyperparameter tuning — default model parameters used
2.No threshold optimisation for low-chance recall
3.Single dataset — no external validation cohort tested
4.Partner features limited to semen analysis result only
5.Cross-sectional snapshot — no longitudinal follow-up data
```

### Future Work
```
1.Hyperparameter tuning (RandomizedSearchCV / Optuna)
2.Advanced imputation (KNN Imputer, MICE)
3.Decision threshold optimisation for low-chance class
4. External validation on independent patient cohorts
5. Web or mobile deployment (FastAPI + React frontend)
6. EHR integration for clinical settings
7. Extended biomarker panel (LH, prolactin, estradiol)
8.A/B Testing
```

---

##  Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-00A86B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/imbalanced--learn-SMOTE-9C27B0?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
</p>

---

## Author

**Neha Reddy Ramidi**
- [MS in Data science / Institution]
- Gmail:[rnehareddy74@gmail.com]
-  https://www.linkedin.com/in/neha-reddy-43a0a4243/

---

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgements

- SHAP — Lundberg & Lee (2017), *"A Unified Approach to Interpreting Model Predictions"*, NeurIPS
- SMOTE — Chawla et al. (2002), *"SMOTE: Synthetic Minority Over-sampling Technique"*, JAIR
- Scikit-learn — Pedregosa et al. (2011), *"Scikit-learn: Machine Learning in Python"*, JMLR

---

<p align="center">
  <i>This Project(FertiPredict) is intended for early-stage pre-screening purposes only.<br>
  It is not a medical device and does not replace clinical diagnosis or specialist advice.<br>
  Always consult a qualified reproductive endocrinologist for medical decisions.</i>
</p>

<p align="center">Made with 💚 for reproductive health awareness</p>

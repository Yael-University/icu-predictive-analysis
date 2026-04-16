# ICU Predictive Analysis - Patient Outcome Prediction

**Dual ML system predicting ICU length of stay and 30-day readmission risk using MIMIC-IV dataset (545K admissions). Achieves 70% accuracy within 3 days for LOS predictions and 68% AUC for readmission risk.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV-red.svg)](https://mimic.mit.edu/)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Clinical Impact](#clinical-impact)
- [Future Improvements](#future-improvements)

---

## Overview

This healthcare ML system predicts two critical ICU patient outcomes:

1. **Length of Stay (LOS) Prediction** - Regression model forecasting hospital stay duration
2. **30-Day Readmission Risk** - Classification model identifying high-risk patients

Built on the prestigious **MIMIC-IV dataset** (Medical Information Mart for Intensive Care), the system processes **545,847 hospital admissions** with 30+ clinical features to enable proactive care management and resource optimization.

---

## Problem Statement

Healthcare providers face critical challenges in patient care planning:

- **How long will patients stay?** Accurate LOS predictions optimize bed allocation and staffing
- **Who will be readmitted?** Early identification enables preventive interventions
- **How to allocate resources?** Risk stratification guides intensive case management

This system addresses these challenges through:
- **Predictive modeling** using demographic, clinical, and administrative data
- **Feature engineering** from ICD diagnosis codes and medication records
- **Class balancing** techniques for imbalanced readmission data (33% positive class)

---

## Key Results

### Length of Stay Prediction (Ridge Regression)

| Metric | Training | Testing |
|--------|----------|---------|
| **MAE** | 2.851 days | 2.861 days |
| **RMSE** | 5.510 days | 5.636 days |
| **R² Score** | 0.4072 | 0.3948 |

**Prediction Accuracy:**
- Within ±1 day: **33.57%** of predictions
- Within ±2 days: **55.79%** of predictions
- Within ±3 days: **70.30%** of predictions

**Key Insight:** Consistent train/test performance indicates minimal overfitting and reliable generalization.

### 30-Day Readmission Risk (Gradient Boosting)

| Metric | Value |
|--------|-------|
| **Accuracy** | 69.43% |
| **AUC-ROC** | 0.683 |
| **Precision (Readmitted)** | 60% |
| **Recall (Readmitted)** | 23% |
| **F1-Score** | 0.33 |

**High-Risk Patient Identification:**
- Flagged as high-risk (>50% probability): **7,903 patients**
- Actual readmissions captured: **4,722 patients**
- **Precision in high-risk group: 59.75%**

**Key Insight:** Model balances false positives (unnecessary interventions) with false negatives (missed at-risk patients), optimized for clinical workflow.

---

## Tech Stack

**Core Technologies:**
- **Python 3.11+** - Primary language
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - ML pipeline & preprocessing

**Machine Learning:**
- **Ridge Regression (α=1.0)** - Length of stay prediction
- **Gradient Boosting Classifier** - Readmission risk (100 trees, lr=0.1)
- **StandardScaler** - Feature normalization
- **5-Fold Cross-Validation** - Model robustness

**Data Source:**
- **MIMIC-IV** - De-identified ICU patient data from Beth Israel Deaconess Medical Center

**Visualization:**
- **Matplotlib & Seaborn** - ROC curves, confusion matrices, feature importance

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   MIMIC-IV Data Pipeline                     │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  MIMIC-IV Raw Data (545K admissions)                         │
│  - admissions.csv                                            │
│  - patients.csv                                              │
│  - diagnoses_icd.csv                                         │
│  - procedures_icd.csv                                        │
│  - prescriptions.csv (optional)                              │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Data Preprocessing                                          │
│  - Merge tables on hadm_id (admission ID)                   │
│  - Exclude: deaths, last admissions (no follow-up)          │
│  - Create readmission flag (30-day window)                  │
│  - Extract ICD code groups (top 15 diagnostic categories)   │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Feature Engineering (30+ features)                          │
│  Demographics: age, gender, race, language, marital_status   │
│  History: num_prev_admissions, days_since_last_admission     │
│  Current: LOS, admission_type, ED_stay, num_diagnoses       │
│  Clinical: procedure_count, medication_count, ICD_groups     │
│  Discharge: discharge_location, insurance_type               │
└──────┬───────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────┬────────────────────────────┐
       ▼                             ▼                            ▼
┌──────────────┐          ┌──────────────────┐      ┌──────────────────┐
│  Pipeline 1: │          │  Imputation:     │      │  Pipeline 2:     │
│  Numeric     │──────────│  - Numeric:      │      │  Categorical     │
│  Features    │          │    Median        │      │  Features        │
│              │          │  - Categorical:  │      │                  │
│              │          │    Most Frequent │      │                  │
└──────┬───────┘          └──────────────────┘      └──────┬───────────┘
       │                                                      │
       │                  ┌──────────────────┐               │
       └──────────────────│  StandardScaler  │───────────────┘
                          └────────┬─────────┘
                                   │
                   ┌───────────────┴──────────────┐
                   ▼                              ▼
          ┌─────────────────┐          ┌──────────────────┐
          │  Ridge           │          │  Gradient        │
          │  Regression      │          │  Boosting        │
          │  (LOS)           │          │  (Readmission)   │
          └────────┬─────────┘          └────────┬─────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │  Evaluation & Outputs  │
                     │  - ROC curves          │
                     │  - Confusion matrices  │
                     │  - Feature importance  │
                     │  - Risk distribution   │
                     └────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.11 or higher
- MIMIC-IV database access (requires credentialed access from PhysioNet)
- 8GB+ RAM

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/icu-predictive-analysis.git
cd icu-predictive-analysis
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Obtain MIMIC-IV data:**
- Complete required training at [PhysioNet](https://physionet.org/)
- Sign data use agreement
- Download MIMIC-IV CSV files
- Place in `MIMIC/` directory

---

## Usage

### Train Length of Stay Model

```bash
python length_of_stay_prediction.py
```

**Output:**
- `los_evaluation.png` - Prediction accuracy plots
- `los_feature_importance.png` - Top contributing features
- Model metrics printed to console

### Train Readmission Risk Model

```bash
python readmission_risk_prediction.py
```

**Output:**
- `readmission_roc_curve.png` - ROC curve (AUC=0.683)
- `readmission_confusion_matrix.png` - Classification performance
- `readmission_feature_importance.png` - Top 20 features
- `readmission_risk_distribution.png` - Probability distributions
- `readmission_results.txt` - Full classification report

### Expected Runtime
- **Data preprocessing**: ~30 minutes (545K admissions)
- **LOS model training**: ~10 minutes
- **Readmission model training**: ~15 minutes
- **Total pipeline**: ~1 hour

---

## Model Details

### Model 1: Length of Stay Prediction

**Algorithm:** Ridge Regression (L2 Regularized Linear Model)

**Input Features (12):**
- Demographics: Age, Gender
- Admission: Emergency flag, admission hour/weekday
- ED Metrics: ED visit flag, ED length of stay
- Clinical: Number of diagnoses, number of procedures
- Administrative: Insurance type, hospital mortality flag

**Why Ridge Regression?**
- Handles multicollinearity among features
- Fast training and prediction (real-time capable)
- Interpretable coefficients for clinical validation
- Regularization (α=1.0) prevents overfitting

**Performance Characteristics:**
- Best for short-to-medium stays (1-10 days)
- 70% of predictions within 3-day margin
- Lower accuracy for extended stays (>20 days)

### Model 2: 30-Day Readmission Risk

**Algorithm:** Gradient Boosting Classifier

**Input Features (30+):**
- Demographics: Age, gender, race, language, marital status
- Admission History: Previous admissions, days since last admission
- Current Stay: LOS, admission type, emergency flag
- Clinical Complexity: Diagnosis count, procedure count, ICD groups
- Medications: Medication count, unique medications
- Discharge: Discharge location, insurance

**Why Gradient Boosting?**
- Captures complex non-linear relationships
- Handles mixed data types (numeric + categorical)
- Provides probability scores for risk stratification
- Feature importance reveals clinical drivers

**Top 10 Predictive Features:**
1. Days Since Last Admission (27.70%)
2. Number of Previous Admissions (15.15%)
3. Number of Medications (9.20%)
4. Length of Stay (9.17%)
5. Age (3.89%)
6. ICD F (Mental/Behavioral) Diagnoses (2.70%)
7. Discharge to Psych Facility (2.29%)
8. Number of Unique Medications (2.04%)
9. ED Length of Stay (1.77%)
10. Urgent Admission Type (1.67%)

**Clinical Interpretation:**
- Recent admissions are strongest predictor (recency effect)
- Polypharmacy indicates complex comorbidities
- Mental health diagnoses increase readmission risk
- ED utilization patterns matter

---

## Clinical Impact

### Use Cases

**1. Discharge Planning**
- Identify high-risk patients requiring enhanced support
- Allocate social work/case management resources
- Schedule early follow-up appointments

**2. Care Coordination**
- Trigger medication reconciliation for high-risk patients
- Arrange post-discharge phone calls
- Coordinate with primary care providers

**3. Resource Allocation**
- Optimize bed turnover planning with LOS predictions
- Focus intensive case management on top-risk quartile
- Prevent unnecessary readmissions (cost savings)

**4. Quality Improvement**
- Track and reduce preventable readmissions
- Benchmark against risk-adjusted targets
- Identify systemic care gaps

### Clinical Workflow Integration

```
Patient Admitted → ICU → Model Runs → Risk Score Generated
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
                Low Risk (<30%)                          High Risk (>50%)
                Standard discharge                        Enhanced interventions:
                                                         - Case management
                                                         - Medication review
                                                         - Early follow-up
                                                         - Post-discharge calls
```

---

## Future Improvements

### Short-term
- [ ] Add SHAP values for individual patient explanations
- [ ] Implement time-series features (vitals trends)
- [ ] Create Streamlit dashboard for clinical use
- [ ] Add model retraining pipeline

### Medium-term
- [ ] Integrate real-time EHR data feeds
- [ ] Deploy as FHIR-compatible API
- [ ] Add survival analysis (time-to-readmission)
- [ ] Multi-hospital validation study

### Long-term
- [ ] Deep learning with attention mechanisms
- [ ] Incorporate clinical notes (NLP)
- [ ] Causal inference for interventions
- [ ] Mobile app for care teams

---

## Data Privacy & Ethics

- All data is **de-identified** per HIPAA guidelines
- Patient consent obtained for MIMIC-IV research use
- Model outputs are **decision support only** - not autonomous decisions
- Regular bias audits for demographic fairness
- Transparent model limitations documented

---

## License

This project is licensed under the MIT License. MIMIC-IV data usage requires separate PhysioNet credentialing.

---

## Acknowledgments

- **MIMIC-IV Dataset:** Johnson, A., et al. (2023). MIMIC-IV Clinical Database v2.2.
- **Institution:** Beth Israel Deaconess Medical Center, Boston
- **Funding:** Supported by NIH R01 grant (MIMIC project)

---

## Contact

**Yael Mendez**  
- GitHub: [@yaelmendez](https://github.com/yaelmendez)
- LinkedIn: [Yael Mendez](https://linkedin.com/in/yaelmendez)
- Email: your.email@example.com

---

**Built for better patient outcomes**

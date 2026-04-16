# ICU Predictive Analysis Models - Slide Information

## MODEL 1: LENGTH OF STAY PREDICTION

### Overview
- **Algorithm:** Ridge Regression (L2 Regularized Linear Model)
- **Objective:** Predict hospital length of stay in days for ICU patients
- **Data Source:** MIMIC-IV Database
- **Dataset Size:** 545,847 admissions (after preprocessing)

### Model Architecture
- **Algorithm Type:** Supervised Learning - Regression
- **Model:** Ridge Regression with α = 1.0
- **Feature Scaling:** StandardScaler (normalized features)
- **Train/Test Split:** 80/20 (436,677 training / 109,170 testing)
- **Cross-Validation:** 5-fold CV

### Input Features (12 features)
**Demographics:**
- Age
- Gender (IS_MALE)

**Admission Characteristics:**
- Emergency admission flag
- Admission hour
- Admission day of week

**Emergency Department Metrics:**
- Has ED visit flag
- ED length of stay (hours)

**Clinical Complexity:**
- Number of conditions/diagnoses
- Number of procedures

**Administrative:**
- Insurance type (Medicare/Medicaid flags)
- Hospital mortality flag

### Performance Metrics

**Training Set:**
- Mean Absolute Error (MAE): **2.851 days**
- Root Mean Squared Error (RMSE): **5.510 days**
- R² Score: **0.4072** (40.72% variance explained)

**Test Set:**
- Mean Absolute Error (MAE): **2.861 days**
- Root Mean Squared Error (RMSE): **5.636 days**
- R² Score: **0.3948** (39.48% variance explained)
- Mean Absolute Percentage Error: 161.79%

**Cross-Validation:**
- 5-fold CV MAE: **2.869 ± 0.012 days**

### Prediction Accuracy
- **Within ±1 day: 33.57%** of predictions
- **Within ±2 days: 55.79%** of predictions
- **Within ±3 days: 70.30%** of predictions

### Data Characteristics
- LOS Range: 0.0 - 321.6 days
- Mean LOS: 4.76 days
- Median LOS: 2.82 days

### Key Insights
✅ **Model Strengths:**
- Consistent performance (training vs test metrics are similar - minimal overfitting)
- Over 70% of predictions within 3 days of actual LOS
- Fast training and prediction
- Interpretable coefficients

⚠️ **Considerations:**
- High MAPE due to short actual LOS values (small denominator effect)
- Better performance for shorter stays than extended stays
- R² of ~0.40 indicates moderate predictive power

### Output Visualizations
1. **Predicted vs Actual Scatter Plot** - Shows prediction accuracy
2. **Residual Plot** - Shows error distribution
3. **Error Distribution Histogram** - Mean and median error analysis
4. **Performance by LOS Range** - Model accuracy across different stay lengths
5. **Feature Importance Chart** - Top contributing features

---

## MODEL 2: 30-DAY READMISSION RISK PREDICTION

### Overview
- **Algorithm:** Gradient Boosting Classifier
- **Objective:** Predict probability of hospital readmission within 30 days of discharge
- **Data Source:** MIMIC-IV Database
- **Target Variable:** Binary (Readmitted = 1, Not Readmitted = 0)

### Model Architecture
- **Algorithm Type:** Supervised Learning - Classification
- **Model:** Gradient Boosting Classifier
  - n_estimators: 100 trees
  - learning_rate: 0.1
  - max_depth: 5
- **Train/Test Split:** 80/20 (stratified sampling)
- **Preprocessing Pipeline:**
  - Numeric features: Median imputation + StandardScaler
  - Categorical features: Most frequent imputation + One-Hot Encoding

### Input Features (30+ features across 6 categories)

**1. Demographics:**
- Age
- Gender
- Race
- Language
- Marital status

**2. Admission History:**
- Number of previous admissions
- Days since last admission
- Admission number (chronological)

**3. Current Admission Characteristics:**
- Length of stay (days)
- Admission type (Emergency/Elective/etc.)
- Admission location
- Emergency admission flag
- Elective admission flag

**4. Emergency Department Utilization:**
- Has ED visit flag
- ED length of stay (hours)

**5. Clinical Complexity:**
- Number of diagnoses/conditions
- Number of unique diagnoses
- Number of procedures
- Number of unique procedures
- ICD code group counts (top 15 diagnostic categories)
- Number of medications (if available)
- Number of unique medications (if available)

**6. Discharge Information:**
- Discharge location (strong predictor)
- Insurance type

### Data Preprocessing
**Exclusions:**
- Patients who died during hospital stay (cannot be readmitted)
- Last admission for each patient (no follow-up data)
- Admissions missing critical features

**Readmission Definition:**
- Patient readmitted within 30 days of discharge
- Days to readmission > 0 and ≤ 30

### Performance Metrics

**Dataset Statistics:**
- Total admissions analyzed: 316,031
- Readmissions within 30 days: 104,302
- **Readmission rate: 33.00%**
- Test set size: 63,207 admissions

**Overall Performance:**
- **Accuracy: 69.43%**
- **AUC-ROC: 0.683**

**Classification Metrics (Readmitted Class):**
- **Precision: 60%** (of patients predicted to be readmitted, 60% actually were)
- **Recall: 23%** (model identifies 23% of actual readmissions)
- **F1-Score: 0.33**

**Classification Metrics (Not Readmitted Class):**
- Precision: 71%
- Recall: 92%
- F1-Score: 0.80

**Confusion Matrix:**
```
                    Predicted
                Not Readmit  Readmit
Actual Not      39,165       3,181
Actual Readmit  16,139       4,722
```

**High-Risk Patient Identification (>50% probability threshold):**
- Patients flagged as high-risk: 7,903
- Actual readmissions captured: 4,722
- **Precision in high-risk group: 59.75%**

### Clinical Application

**High-Risk Patient Identification:**
- Patients flagged with >50% readmission probability
- Enables targeted intervention and care management
- Supports resource allocation for follow-up care

**Use Cases:**
1. **Discharge Planning** - Enhanced support for high-risk patients
2. **Care Coordination** - Early follow-up appointments
3. **Resource Allocation** - Focus intensive case management
4. **Quality Improvement** - Track and reduce preventable readmissions

### Output Visualizations
1. **Confusion Matrix Heatmap** - Classification performance
2. **ROC Curve** - Discrimination ability with AUC score
3. **Risk Probability Distribution** - Readmitted vs Not Readmitted groups
4. **Feature Importance** - Top 20 most influential features
5. **High-Risk Analysis** - Performance in identifying at-risk patients

### Key Insights
✅ **Model Strengths:**
- Captures complex non-linear relationships (Gradient Boosting)
- Handles mixed data types (numeric + categorical)
- Provides probability scores for risk stratification
- Feature importance reveals clinical drivers of readmission

⚠️ **Clinical Considerations:**
- Class imbalance (fewer readmissions than non-readmissions)
- Balance between false positives (unnecessary interventions) and false negatives (missed at-risk patients)
- Model should complement, not replace, clinical judgment

---

## COMPARISON: RIDGE REGRESSION vs GRADIENT BOOSTING

### When to Use Each:

**Ridge Regression (Length of Stay):**
- ✅ Continuous numeric prediction
- ✅ Fast training and inference
- ✅ Highly interpretable
- ✅ Works well with linear relationships
- ✅ Handles multicollinearity

**Gradient Boosting (Readmission Risk):**
- ✅ Binary classification tasks
- ✅ Captures complex non-linear patterns
- ✅ Handles feature interactions automatically
- ✅ Robust to outliers
- ✅ Superior performance on complex tasks (typically)

---

## DATA SOURCE: MIMIC-IV

**Medical Information Mart for Intensive Care (MIMIC-IV)**
- De-identified health data from ICU patients
- Beth Israel Deaconess Medical Center, Boston
- Contains demographics, vital signs, lab tests, medications, diagnoses, procedures
- Widely used benchmark dataset for healthcare ML research

**Tables Used:**
- `admissions.csv` - Hospital admission records
- `patients.csv` - Patient demographics
- `diagnoses_icd.csv` - ICD diagnosis codes
- `procedures_icd.csv` - ICD procedure codes
- `prescriptions.csv` - Medication data (optional)

---

## TECHNICAL IMPLEMENTATION

**Programming Language:** Python 3
**Key Libraries:**
- scikit-learn (machine learning models)
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib (visualization)
- seaborn (statistical visualization)

**Reproducibility:**
- Random state: 42 (for consistent results)
- Version-controlled codebase
- Standardized preprocessing pipeline

---

### Top 10 Most Important Predictive Features
1. **Days Since Last Admission** (27.70%) - Strongest predictor
2. **Number of Previous Admissions** (15.15%)
3. **Number of Medications** (9.20%)
4. **Length of Stay** (9.17%)
5. **Age** (3.89%)
6. **ICD F (Mental/Behavioral) Diagnoses Count** (2.70%)
7. **Discharge to Psych Facility** (2.29%)
8. **Number of Unique Medications** (2.04%)
9. **ED Length of Stay** (1.77%)
10. **Urgent Admission Type** (1.67%)

---

*Document Generated: March 31, 2026*
*Status: ✅ All models completed successfully*

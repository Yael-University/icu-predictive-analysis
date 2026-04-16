# ICU Predictive Analysis

This project uses synthetic patient data to predict ICU patient outcomes, specifically **mortality**, using a simple machine learning pipeline (for now). The dataset is generated with [Synthea](https://github.com/synthetichealth/synthea), which creates realistic but synthetic patient records. MIMIC-IV Integration will occur at a later time. We extract key demographic features (age, gender, race) and the latest available vital signs (BMI, systolic/diastolic blood pressure, and heart rate) for each patient. These features are preprocessed using imputation (to handle missing values), scaling (for numeric features), and one-hot encoding (for categorical features). A logistic regression model with class balancing is then trained to distinguish between patients who survived versus those who died. Model performance is evaluated using precision, recall, F1-score, and a confusion matrix to show true/false positives and negatives.

---

## Overview

The ML pipeline works as follows:

1. **Data preprocessing**: Patient demographics and last recorded vitals are extracted from Synthea CSV files. Missing numeric values are imputed with the mean, and categorical variables are one-hot encoded.
2. **Feature selection**: Key features include age, gender, race, BMI, blood pressure, and heart rate.
3. **Modeling**: Logistic Regression is trained on the processed features to predict the binary mortality outcome.
4. **Evaluation**: The model is evaluated with a classification report and confusion matrix.

---

## Setup Synthea (First clone this repo, then proceed with step 1)

1. Clone the Synthea repository:

```bash
git clone https://github.com/synthetichealth/synthea.git
cd synthea
```
Generate patient data (adjust the number of patients as needed):

```bash
./run_synthea -p 100
```

This will create synthetic patient records in synthea/output/csv/.

In your ML project, point to the CSV folder (should already be done):

```python
csv_path = "synthea/output/csv/"
```
NOTE: Do not commit the full Synthea repository or generated CSVs to GitHub (PLEASE). Add the following to .gitignore:

```bash
synthea/output/
*.csv
```

Usage
Run the main script:

```bash
python icu-predictive-analysis.py
```
The script will:

Load the patient CSVs

Process key features

Train a Logistic Regression model

Display a classification report and confusion matrix

Dependencies
Python 3.11+

pandas

scikit-learn

Install dependencies via:

```bash
pip install -r requirements.txt
```

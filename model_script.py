# This base file uses logistic regression to predict patient mortality based on Synthea-generated EHR data
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Import custom S3 helper
from s3_utils import save_and_upload_plot

# Path to synthea folder
csv_path = "synthea/output/csv/"

# Load main CSVs from Synthea (using just some, we can add more later for better accuracy)
patients = pd.read_csv(csv_path + "patients.csv")
conditions = pd.read_csv(csv_path + "conditions.csv")
observations = pd.read_csv(csv_path + "observations.csv")
encounters = pd.read_csv(csv_path + "encounters.csv")

print("Observation columns:", observations.columns.tolist())
print(observations.head())

print("Patients:", patients.shape)
print("Conditions:", conditions.shape)
print("Observations:", observations.shape)
print("Encounters:", encounters.shape)

# --- Add Mortality flag ---------------------------------------------------------------------------------------------
patients["MORTALITY"] = patients["DEATHDATE"].notnull().astype(int)
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
patients["AGE"] = (datetime.now() - patients["BIRTHDATE"]).dt.days // 365

print(patients[["Id", "BIRTHDATE", "DEATHDATE", "MORTALITY", "AGE"]].head())

# --- Take the last observation per patient for key vitals ------------------------------------------------------------
key_vitals = [
    "Body mass index (BMI) [Ratio]",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate"
]

# Keep only the above vital rows
obs_filtered = observations[observations["DESCRIPTION"].isin(key_vitals)]

# Sort by DATE and keep the last observation per patient per vital
obs_latest = (
    obs_filtered.sort_values("DATE")
    .groupby(["PATIENT", "DESCRIPTION"])
    .last()
    .reset_index()
)

# Pivot to wide format (one row per patient, vitals as columns)
obs_wide = (
    obs_latest.pivot(index="PATIENT", columns="DESCRIPTION", values="VALUE")
    .reset_index()
)

# Remove multi-index column naming
obs_wide.columns.name = None

print("Vitals (wide):")
print(obs_wide.head())

# --- Merge with patients ---
data = patients.merge(obs_wide, left_on="Id", right_on="PATIENT", how="left")

# --- Features and target ---
features = ["AGE", "GENDER", "RACE"] + key_vitals
X = data[features]
y = data["MORTALITY"]

print(X.head())
print(y.head())

# --- Train And Test -----------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
numeric_features = ["AGE"] + key_vitals
categorical_features = ["GENDER", "RACE"]

# Imputer to fix NaN values with the mean (other option is to drop patient data)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Logistic regression pipeline with class balancing
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Train
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ------------------------------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Alive", "Dead"], yticklabels=["Alive", "Dead"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")

# Save and upload to S3
bucket_name = "asghar-model-output"
url = save_and_upload_plot(plt, bucket_name, filename="logreg_confusion_matrix.png")

print("Model training and evaluation complete.")
print(f"View confusion matrix: {url}")

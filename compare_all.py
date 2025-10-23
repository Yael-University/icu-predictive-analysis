import boto3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
from datetime import datetime
from s3_utils import save_and_upload_plot, upload_file_to_s3

# --- Configuration ---
bucket_name = "asghar-model-output"
base_prefix = "ml_outputs"
comparison_folder = "comparison"

# --- Initialize S3 client ---
s3 = boto3.client("s3")

# --- Helper: Read metrics_summary.txt from S3 ---
def read_metrics_from_s3(model_name):
    key = f"{base_prefix}/{model_name}/metrics_summary.txt"
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        text = response["Body"].read().decode("utf-8")
        lines = text.strip().split("\n")
        metrics = {"Model": model_name.replace("_", " ").title()}
        for line in lines:
            if ":" in line:
                name, value = line.split(":", 1)
                name = name.strip()
                try:
                    metrics[name] = float(value.strip())
                except ValueError:
                    metrics[name] = value.strip()
        return metrics
    except s3.exceptions.NoSuchKey:
        print(f"⚠️ No metrics found for {model_name}. Skipping.")
    except Exception as e:
        print(f"Error reading {key}: {e}")
    return None

# --- Discover all model folders in ml_outputs ---
def discover_model_folders():
    try:
        paginator = s3.get_paginator("list_objects_v2")
        result = paginator.paginate(Bucket=bucket_name, Prefix=base_prefix + "/")
        models = set()
        for page in result:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # look for metrics_summary.txt and infer folder name
                if key.endswith("metrics_summary.txt"):
                    parts = key.split("/")
                    if len(parts) >= 3:
                        model_folder = parts[1]
                        models.add(model_folder)
        return sorted(list(models))
    except Exception as e:
        print(f"Error discovering models: {e}")
        return []

# --- Read metrics for all discovered models ---
models = discover_model_folders()
if not models:
    print("No model folders found in S3 (ml_outputs/*/metrics_summary.txt). Exiting.")
    exit(1)

records = []
for model in models:
    m = read_metrics_from_s3(model)
    if m:
        records.append(m)

if not records:
    print("No metrics found for any models.")
    exit(1)

# --- Build DataFrame ---
df = pd.DataFrame(records)
# Normalize column names just in case
columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
df = df[[c for c in columns if c in df.columns]]

# --- Print Summary ---
print("\n=== Model Comparison Summary ===")
print(df.to_string(index=False))
print("================================\n")

# --- Save CSV and upload ---
csv_filename = "metrics_comparison.csv"
df.to_csv(csv_filename, index=False)
upload_file_to_s3(csv_filename, bucket_name, f"{comparison_folder}/{csv_filename}")
os.remove(csv_filename)

# --- Visualization ---
melted = df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(x="Metric", y="Score", hue="Model", data=melted)
plt.title("Model Comparison Across All Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title="Model", loc="lower right")

save_and_upload_plot(
    plt,
    bucket_name,
    folder=comparison_folder,
    filename="comparison_chart.png"
)

print(f"Comparison complete. Results uploaded to s3://{bucket_name}/{comparison_folder}/")

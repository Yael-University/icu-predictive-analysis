import os
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "MIMIC"

ADMISSIONS_CSV = os.path.join(DATA_DIR, "admissions.csv")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients.csv")
DIAGNOSES_CSV = os.path.join(DATA_DIR, "diagnoses_icd.csv")
PROCEDURES_CSV = os.path.join(DATA_DIR, "procedures_icd.csv")

RANDOM_STATE = 42

def must_exist(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

def safe_read_csv(path: str, usecols=None, parse_dates=None, dtype=None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=False,
    )

def evaluate_model(y_train, y_pred_train, y_test, y_pred_test):
    """Comprehensive model evaluation"""
    
    # Clip negative predictions
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_test = np.maximum(y_pred_test, 0)
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    
    # Test metrics
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    
    # Accuracy within tolerance ranges
    tolerance_1 = np.mean(np.abs(y_test - y_pred_test) <= 1) * 100
    tolerance_2 = np.mean(np.abs(y_test - y_pred_test) <= 2) * 100
    tolerance_3 = np.mean(np.abs(y_test - y_pred_test) <= 3) * 100
    
    print(f"\n{'='*80}")
    print(f"XGBOOST REGRESSOR - LENGTH OF STAY PREDICTION (Days)")
    print(f"{'='*80}")
    
    print(f"\nTraining Set Performance:")
    print(f"  Mean Absolute Error (MAE):  {train_mae:.3f} days")
    print(f"  Root Mean Squared Error:    {train_rmse:.3f} days")
    print(f"  R² Score:                   {train_r2:.4f}")
    
    print(f"\nTest Set Performance:")
    print(f"  Mean Absolute Error (MAE):  {test_mae:.3f} days")
    print(f"  Root Mean Squared Error:    {test_rmse:.3f} days")
    print(f"  R² Score:                   {test_r2:.4f}")
    print(f"  Mean Absolute % Error:      {test_mape:.2f}%")
    
    print(f"\nPrediction Accuracy (Test Set):")
    print(f"  Within ±1 day:              {tolerance_1:.2f}%")
    print(f"  Within ±2 days:             {tolerance_2:.2f}%")
    print(f"  Within ±3 days:             {tolerance_3:.2f}%")
    
    return y_pred_test

def plot_results(y_test, y_pred_test):
    """Generate visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.4, s=10)
    max_val = max(y_test.max(), y_pred_test.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual LOS (days)', fontsize=11)
    axes[0, 0].set_ylabel('Predicted LOS (days)', fontsize=11)
    axes[0, 0].set_title('Predicted vs Actual Length of Stay', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Residuals Plot
    residuals = y_test - y_pred_test
    axes[0, 1].scatter(y_pred_test, residuals, alpha=0.4, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted LOS (days)', fontsize=11)
    axes[0, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Error Distribution
    errors = np.abs(y_test - y_pred_test)
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].axvline(errors.mean(), color='r', linestyle='--', lw=2, 
                       label=f'Mean: {errors.mean():.2f} days')
    axes[1, 0].axvline(np.median(errors), color='g', linestyle='--', lw=2, 
                       label=f'Median: {np.median(errors):.2f} days')
    axes[1, 0].set_xlabel('Absolute Error (days)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Performance by LOS Range
    los_bins = [0, 3, 7, 14, 365]
    los_labels = ['≤3 days', '4-7 days', '8-14 days', '>14 days']
    y_test_binned = pd.cut(y_test, bins=los_bins, labels=los_labels)
    
    bin_mae = []
    bin_counts = []
    for label in los_labels:
        mask = y_test_binned == label
        if mask.sum() > 0:
            bin_mae.append(mean_absolute_error(y_test[mask], y_pred_test[mask]))
            bin_counts.append(mask.sum())
        else:
            bin_mae.append(0)
            bin_counts.append(0)
    
    bars = axes[1, 1].bar(los_labels, bin_mae, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 1].set_xlabel('Actual LOS Range', fontsize=11)
    axes[1, 1].set_ylabel('Mean Absolute Error (days)', fontsize=11)
    axes[1, 1].set_title('Model Performance by LOS Range', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'n={count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('los_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Evaluation plots saved as 'los_evaluation.png'")

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance (works for XGBoost and Ridge)."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.abs(model.coef_)
    
    # Adjust top_n if we have fewer features
    top_n = min(top_n, len(importance))
    
    # Get top features
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices][::-1], color='teal', edgecolor='black')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]], fontsize=9)
    plt.xlabel('Absolute Coefficient Value', fontsize=11)
    plt.title(f'Top {top_n} Most Important Features - XGBoost Regressor',
             fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('los_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Feature importance saved as 'los_feature_importance.png'")

def main() -> None:
    must_exist(ADMISSIONS_CSV)
    must_exist(PATIENTS_CSV)
    must_exist(DIAGNOSES_CSV)
    must_exist(PROCEDURES_CSV)

    print("="*80)
    print("LENGTH OF STAY PREDICTION - SIMPLE RIDGE REGRESSION MODEL")
    print("="*80)
    
    print("\n[1/5] Loading data...")
    admissions = safe_read_csv(
        ADMISSIONS_CSV,
        usecols=[
            "subject_id", "hadm_id", "admittime", "dischtime",
            "admission_type", "insurance", "race",
            "edregtime", "edouttime", "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime", "edregtime", "edouttime"],
        dtype={"subject_id": "int32", "hadm_id": "int32"},
    )

    patients = safe_read_csv(
        PATIENTS_CSV,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        dtype={"subject_id": "int32", "anchor_age": "float32", "anchor_year": "int32"},
    )

    dx = safe_read_csv(
        DIAGNOSES_CSV,
        usecols=["hadm_id", "icd_code"],
        dtype={"hadm_id": "int32", "icd_code": "string"},
    )

    px = safe_read_csv(
        PROCEDURES_CSV,
        usecols=["hadm_id", "icd_code"],
        dtype={"hadm_id": "int32", "icd_code": "string"},
    )

    print(f"   ✓ Loaded {admissions.shape[0]:,} admissions")

    # Calculate Length of Stay
    print("\n[2/5] Processing features...")
    admissions = admissions.dropna(subset=["dischtime", "admittime"])
    admissions["LOS_DAYS"] = (
        (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400
    ).astype("float32")
    
    # Remove outliers
    admissions = admissions[(admissions["LOS_DAYS"] > 0) & (admissions["LOS_DAYS"] <= 365)]
    
    print(f"   ✓ LOS range: {admissions['LOS_DAYS'].min():.1f} - {admissions['LOS_DAYS'].max():.1f} days")
    print(f"   ✓ Mean LOS: {admissions['LOS_DAYS'].mean():.2f} days, Median: {admissions['LOS_DAYS'].median():.2f} days")

    # Simple feature engineering
    admissions["ADMIT_HOUR"] = admissions["admittime"].dt.hour.astype("int32")
    admissions["ADMIT_DAY_OF_WEEK"] = admissions["admittime"].dt.dayofweek.astype("int32")
    
    base = admissions.merge(patients, on="subject_id", how="left")
    
    # Age
    base["ADMIT_YEAR"] = base["admittime"].dt.year.astype("int32")
    base["AGE"] = (base["anchor_age"] + (base["ADMIT_YEAR"] - base["anchor_year"])).astype("float32")
    
    # ED visit features
    base["ED_LOS_HOURS"] = ((base["edouttime"] - base["edregtime"]).dt.total_seconds() / 3600).astype("float32")
    base["HAS_ED_VISIT"] = base["edregtime"].notna().astype("int8")
    
    # Simple flags
    base["IS_EMERGENCY"] = (base["admission_type"] == "EMERGENCY").astype("int8")
    base["DIED_IN_HOSPITAL"] = base["hospital_expire_flag"].astype("int8")
    base["IS_MALE"] = (base["gender"] == "M").astype("int8")
    
    # Insurance encoding (simple binary for major categories)
    base["HAS_MEDICARE"] = (base["insurance"] == "Medicare").astype("int8")
    base["HAS_MEDICAID"] = (base["insurance"] == "Medicaid").astype("int8")
    
    # Count features (fast aggregation)
    num_conditions = dx.groupby("hadm_id").size().rename("NUM_CONDITIONS")
    num_procedures = px.groupby("hadm_id").size().rename("NUM_PROCEDURES")
    
    # Merge
    feat = base.set_index("hadm_id").join(num_conditions, how="left").join(num_procedures, how="left")
    feat = feat.reset_index()
    
    # Fill missing
    feat["NUM_CONDITIONS"] = feat["NUM_CONDITIONS"].fillna(0).astype("int32")
    feat["NUM_PROCEDURES"] = feat["NUM_PROCEDURES"].fillna(0).astype("int32")
    feat["ED_LOS_HOURS"] = feat["ED_LOS_HOURS"].fillna(0).astype("float32")
    
    print(f"   ✓ Created {feat.shape[1]} features")

    # Select features for model
    feature_cols = [
        "AGE",
        "IS_MALE",
        "HAS_ED_VISIT",
        "ED_LOS_HOURS",
        "IS_EMERGENCY",
        # DIED_IN_HOSPITAL removed — it's a future outcome, not known at admission time.
        # Including it would be data leakage: the model learns "if dead → long stay"
        # but you can't use that signal to predict LOS before it happens.
        "HAS_MEDICARE",
        "HAS_MEDICAID",
        "NUM_CONDITIONS",
        "NUM_PROCEDURES",
        "ADMIT_HOUR",
        "ADMIT_DAY_OF_WEEK",
    ]
    
    # Prepare data
    data = feat[feature_cols + ["LOS_DAYS"]].dropna()
    X = data[feature_cols].values
    y = data["LOS_DAYS"].values
    
    print(f"   ✓ Final dataset: {X.shape[0]:,} samples with {X.shape[1]} features")

    # Log-transform the target before splitting.
    # LOS is right-skewed (most stays are short, a few are very long).
    # Ridge/linear models assume normally distributed residuals and will be
    # dominated by long-stay outliers. log1p compresses those outliers so the
    # model can fit the bulk of cases accurately. We expm1 the predictions
    # back to days after inference.
    y_log = np.log1p(y)

    # Train/test split
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )
    # Keep original-scale labels for evaluation
    _, _, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"   ✓ Training: {X_train.shape[0]:,} samples")
    print(f"   ✓ Testing:  {X_test.shape[0]:,} samples")

    print("\n[4/5] Training XGBoost Regressor model...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train_log)
    print("   ✓ Model trained successfully!")

    # Save model so inference can run without retraining.
    # Predictions need np.expm1() applied: np.expm1(model.predict(X_new))
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/los_model.pkl")
    print("   ✓ Model saved to models/los_model.pkl")

    # Cross-validation (in log space — lower is still better)
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5,
                                scoring='neg_mean_absolute_error')
    print(f"   ✓ 5-fold CV MAE (log scale): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Predictions — inverse-transform from log space back to days
    y_pred_train = np.expm1(model.predict(X_train))
    y_pred_test = np.expm1(model.predict(X_test))
    
    # Evaluate
    print("\n[5/5] Evaluating model...")
    y_pred_test = evaluate_model(y_train_orig, y_pred_train, y_test, y_pred_test)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    plot_results(y_test, y_pred_test)
    plot_feature_importance(model, feature_cols)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nOutput files:")
    print("  • los_evaluation.png         - Model performance visualizations")
    print("  • los_feature_importance.png - Feature importance chart")
    print()

if __name__ == "__main__":
    main()

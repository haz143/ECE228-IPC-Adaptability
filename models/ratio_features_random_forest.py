from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "ratio_feature_random_forest_results"
OUTPUT_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(PROJECT_ROOT / "training_data.csv")
test_df = pd.read_csv(PROJECT_ROOT / "testing_data.csv")

raw_features = [
    "numLoadInsts",
    "numStoreInsts",
    "numInsts",
    "numBranches",
    "intAluAccesses",
    "numOps",
    "L1Icache",
    "L1Dcache",
    "L2cache",
    "pipelinewidth"
]

target = "ipc"


def safe_divide(a, b):
    return a / (b.replace(0, np.nan) + 1e-9)


def add_ratio_features(df):
    df = df.copy()

    df["load_ratio"] = safe_divide(df["numLoadInsts"], df["numInsts"])
    df["store_ratio"] = safe_divide(df["numStoreInsts"], df["numInsts"])
    df["branch_ratio"] = safe_divide(df["numBranches"], df["numInsts"])
    df["microop_ratio"] = safe_divide(df["numOps"], df["numInsts"])
    df["alu_ratio"] = safe_divide(df["intAluAccesses"], df["numOps"])

    df["l1i_per_width"] = safe_divide(df["L1Icache"], df["pipelinewidth"])
    df["l1d_per_width"] = safe_divide(df["L1Dcache"], df["pipelinewidth"])
    df["l2_per_width"] = safe_divide(df["L2cache"], df["pipelinewidth"])
    df["l2_l1d_ratio"] = safe_divide(df["L2cache"], df["L1Dcache"])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


train_ratio = add_ratio_features(train_df)
test_ratio = add_ratio_features(test_df)

ratio_features = [
    "load_ratio",
    "store_ratio",
    "branch_ratio",
    "microop_ratio",
    "alu_ratio",
    "l1i_per_width",
    "l1d_per_width",
    "l2_per_width",
    "l2_l1d_ratio",
    "pipelinewidth"
]


def evaluate_model(name, model, X_train, y_train, X_test, y_test, output_prefix):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    residuals = pred - y_test

    print(f"\n{name}")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    prediction_df = pd.DataFrame({
        "Actual_IPC": y_test,
        "Predicted_IPC": pred,
        "Residual": residuals
    })

    prediction_df.to_csv(OUTPUT_DIR / f"{output_prefix}_predictions.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, pred, alpha=0.7)
    plt.xlabel("Actual IPC")
    plt.ylabel("Predicted IPC")
    plt.title(f"Actual vs Predicted IPC - {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_prefix}_actual_vs_predicted.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(pred, residuals, alpha=0.7)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted IPC")
    plt.ylabel("Residual")
    plt.title(f"Residual Analysis - {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_prefix}_residuals.png", dpi=300)
    plt.close()

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }, model


X_train_raw = train_df[raw_features]
y_train = train_df[target]

X_test_raw = test_df[raw_features]
y_test = test_df[target]

X_train_ratio = train_ratio[ratio_features]
X_test_ratio = test_ratio[ratio_features]

rf_raw = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_ratio = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

raw_result, raw_model = evaluate_model(
    "Original Raw-Feature Random Forest",
    rf_raw,
    X_train_raw,
    y_train,
    X_test_raw,
    y_test,
    "raw_feature_rf"
)

ratio_result, ratio_model = evaluate_model(
    "Ratio-Feature Random Forest",
    rf_ratio,
    X_train_ratio,
    y_train,
    X_test_ratio,
    y_test,
    "ratio_feature_rf"
)

summary = pd.DataFrame([raw_result, ratio_result])
summary.to_csv(OUTPUT_DIR / "ratio_feature_random_forest_summary.csv", index=False)

print("\nSummary:")
print(summary)

plt.figure(figsize=(7, 5))
plt.bar(summary["Model"], summary["RMSE"])
plt.ylabel("RMSE")
plt.title("Raw Features vs Ratio Features")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "raw_vs_ratio_rmse.png", dpi=300)
plt.close()

ratio_importance = pd.DataFrame({
    "Feature": ratio_features,
    "Importance": ratio_model.feature_importances_
}).sort_values("Importance", ascending=False)

ratio_importance.to_csv(OUTPUT_DIR / "ratio_feature_importance.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(ratio_importance["Feature"], ratio_importance["Importance"])
plt.xlabel("Ratio-Based Features")
plt.ylabel("Feature Importance")
plt.title("Feature Importance of Ratio-Based Random Forest")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ratio_feature_importance.png", dpi=300)
plt.close()

print(f"\nSaved all ratio-feature Random Forest results to {OUTPUT_DIR}/")

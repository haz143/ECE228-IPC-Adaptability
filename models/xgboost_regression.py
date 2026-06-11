from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency: xgboost. Install it with `pip install xgboost` "
        "before running this script."
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "xgboost_reproduction_results"
OUTPUT_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(PROJECT_ROOT / "training_data.csv")
test_df = pd.read_csv(PROJECT_ROOT / "testing_data.csv")

features = [
    "numLoadInsts",
    "numStoreInsts",
    "numInsts",
    "numBranches",
    "numOps",
    "L1Icache",
    "L1Dcache",
    "L2cache",
    "pipelinewidth",
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
    "pipelinewidth",
]


def build_xgboost_model():
    return XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def evaluate_model(name, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{name} IPC Prediction")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    return predictions, rmse, mae, r2


def save_prediction_outputs(prefix, y_test, predictions):
    residuals = predictions - y_test

    results = pd.DataFrame(
        {
            "Actual_IPC": y_test,
            "Predicted_IPC": predictions,
            "Residual": residuals,
        }
    )
    results.to_csv(OUTPUT_DIR / f"{prefix}_prediction_results.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel("Actual IPC")
    plt.ylabel("Predicted IPC")
    plt.title("Actual vs. Predicted IPC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_actual_vs_predicted_ipc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(predictions, residuals, alpha=0.7)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Predicted Values")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_residuals_vs_predicted.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.axvline(x=0, linestyle="--")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_residual_histogram.png", dpi=300)
    plt.close()


def save_feature_importance(prefix, model, feature_names):
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    importance_df.to_csv(OUTPUT_DIR / f"{prefix}_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Features")
    plt.ylabel("Feature Importance")
    plt.title("Feature Importance for IPC Prediction")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_feature_importance.png", dpi=300)
    plt.close()


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

xgb = build_xgboost_model()
y_pred, rmse, mae, r2 = evaluate_model(
    "Original XGBoost",
    xgb,
    X_train,
    y_train,
    X_test,
    y_test,
)

save_prediction_outputs("original", y_test, y_pred)
save_feature_importance("original", xgb, features)

metrics = pd.DataFrame(
    {
        "Metric": ["RMSE", "MAE", "R2"],
        "Value": [rmse, mae, r2],
    }
)
metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

# Ratio-feature version: use the same engineered features as the ratio-feature Random Forest.
train_ratio = add_ratio_features(train_df)
test_ratio = add_ratio_features(test_df)

X_train_ratio = train_ratio[ratio_features]
X_test_ratio = test_ratio[ratio_features]

xgb_ratio = build_xgboost_model()
y_pred_ratio, rmse_ratio, mae_ratio, r2_ratio = evaluate_model(
    "\nRatio-Feature XGBoost",
    xgb_ratio,
    X_train_ratio,
    y_train,
    X_test_ratio,
    y_test,
)

save_prediction_outputs("ratio_feature", y_test, y_pred_ratio)
save_feature_importance("ratio_feature", xgb_ratio, ratio_features)

summary = pd.DataFrame(
    {
        "Experiment": ["Original XGBoost", "Ratio-Feature XGBoost"],
        "RMSE": [rmse, rmse_ratio],
        "MAE": [mae, mae_ratio],
        "R2": [r2, r2_ratio],
    }
)
summary.to_csv(OUTPUT_DIR / "reproduction_summary.csv", index=False)

print(f"\nSaved all XGBoost reproduction results to {OUTPUT_DIR}/")

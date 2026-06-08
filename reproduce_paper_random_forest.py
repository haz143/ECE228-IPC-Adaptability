import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("paper_reproduction_results", exist_ok=True)

train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("testing_data.csv")

features = [
    "numLoadInsts",
    "numStoreInsts",
    "numInsts",
    "numBranches",
    "numOps",
    "L1Icache",
    "L1Dcache",
    "L2cache",
    "pipelinewidth"
]

target = "ipc"

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Paper Reproduction: Random Forest IPC Prediction")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

results = pd.DataFrame({
    "Actual_IPC": y_test,
    "Predicted_IPC": y_pred,
    "Residual": y_pred - y_test
})

results.to_csv("paper_reproduction_results/prediction_results.csv", index=False)

metrics = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R2"],
    "Value": [rmse, mae, r2]
})

metrics.to_csv("paper_reproduction_results/metrics.csv", index=False)

# Figure 1: Actual vs Predicted IPC
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual IPC")
plt.ylabel("Predicted IPC")
plt.title("Actual vs. Predicted IPC")
plt.grid(True)
plt.tight_layout()
plt.savefig("paper_reproduction_results/figure1_actual_vs_predicted_ipc.png", dpi=300)
plt.close()

# Figure 3: Residuals vs Predicted Values
residuals = y_pred - y_test

plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.savefig("paper_reproduction_results/figure3_residuals_vs_predicted.png", dpi=300)
plt.close()

# Residual histogram
plt.figure(figsize=(6, 5))
plt.hist(residuals, bins=20, edgecolor="black")
plt.axvline(x=0, linestyle="--")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("paper_reproduction_results/figure3_residual_histogram.png", dpi=300)
plt.close()

# Figure 5: Feature Importance
importance = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values("Importance", ascending=False)

importance_df.to_csv("paper_reproduction_results/feature_importance.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Features")
plt.ylabel("Feature Importance")
plt.title("Feature Importance for IPC Prediction")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("paper_reproduction_results/figure5_feature_importance.png", dpi=300)
plt.close()

# Normalized version: normalize instruction-related features by numInsts
train_norm = train_df.copy()
test_norm = test_df.copy()

norm_features = [
    "numLoadInsts",
    "numStoreInsts",
    "numBranches",
    "numOps"
]

for col in norm_features:
    train_norm[col] = train_norm[col] / train_norm["numInsts"]
    test_norm[col] = test_norm[col] / test_norm["numInsts"]

X_train_norm = train_norm[features]
X_test_norm = test_norm[features]

rf_norm = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

rf_norm.fit(X_train_norm, y_train)
y_pred_norm = rf_norm.predict(X_test_norm)

rmse_norm = np.sqrt(mean_squared_error(y_test, y_pred_norm))
mae_norm = mean_absolute_error(y_test, y_pred_norm)
r2_norm = r2_score(y_test, y_pred_norm)

print("\nNormalized Random Forest IPC Prediction")
print("RMSE:", rmse_norm)
print("MAE:", mae_norm)
print("R2:", r2_norm)

norm_results = pd.DataFrame({
    "Actual_IPC": y_test,
    "Predicted_IPC_Normalized": y_pred_norm,
    "Residual_Normalized": y_pred_norm - y_test
})

norm_results.to_csv("paper_reproduction_results/normalized_prediction_results.csv", index=False)

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_norm, alpha=0.7)
plt.xlabel("Actual IPC")
plt.ylabel("Predicted IPC")
plt.title("Actual vs. Predicted IPC Values")
plt.grid(True)
plt.tight_layout()
plt.savefig("paper_reproduction_results/figure2_normalized_actual_vs_predicted.png", dpi=300)
plt.close()

residuals_norm = y_pred_norm - y_test

plt.figure(figsize=(6, 5))
plt.scatter(y_pred_norm, residuals_norm, alpha=0.7)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.savefig("paper_reproduction_results/figure4_normalized_residuals_vs_predicted.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.hist(residuals_norm, bins=20, edgecolor="black")
plt.axvline(x=0, linestyle="--")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("paper_reproduction_results/figure4_normalized_residual_histogram.png", dpi=300)
plt.close()

summary = pd.DataFrame({
    "Experiment": ["Original Random Forest", "Normalized Random Forest"],
    "RMSE": [rmse, rmse_norm],
    "MAE": [mae, mae_norm],
    "R2": [r2, r2_norm]
})

summary.to_csv("paper_reproduction_results/reproduction_summary.csv", index=False)

print("\nSaved all reproduction results to paper_reproduction_results/")
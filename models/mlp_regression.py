import argparse
import pickle
from pathlib import Path


INPUT_PARAMETERS = [
    "numLoadInsts",
    "numStoreInsts",
    "numInsts",
    "numBranches",
    "intAluAccesses",
    "numOps",
    "model",
    "L1Icache",
    "L1Dcache",
    "L2cache",
    "pipelinewidth",
]

MODEL_FEATURES = [
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

TARGET = "ipc"


def build_features(data):
    import numpy as np
    import pandas as pd

    missing_columns = [col for col in INPUT_PARAMETERS + [TARGET] if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    num_insts = data["numInsts"].replace(0, np.nan)
    num_ops = data["numOps"].replace(0, np.nan)
    pipeline_width = data["pipelinewidth"].replace(0, np.nan)
    l1d_cache = data["L1Dcache"].replace(0, np.nan)

    features = pd.DataFrame(
        {
            "load_ratio": data["numLoadInsts"] / num_insts,
            "store_ratio": data["numStoreInsts"] / num_insts,
            "branch_ratio": data["numBranches"] / num_insts,
            "microop_ratio": data["numOps"] / num_insts,
            "alu_ratio": data["intAluAccesses"] / num_ops,
            "l1i_per_width": data["L1Icache"] / pipeline_width,
            "l1d_per_width": data["L1Dcache"] / pipeline_width,
            "l2_per_width": data["L2cache"] / pipeline_width,
            "l2_l1d_ratio": data["L2cache"] / l1d_cache,
            "pipelinewidth": data["pipelinewidth"],
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return features, data[TARGET]


def train_mlp(train_data_path, test_data_path, output_model_path, predictions_path):
    try:
        import pandas as pd
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing ML dependency. Install pandas and scikit-learn before running this script."
        ) from error

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    x_train, y_train = build_features(train_data)
    x_test, y_test = build_features(test_data)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=50,
        random_state=42,
    )

    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("MLPRegressor IPC prediction")
    print(f"Train data: {train_data_path}")
    print(f"Test data: {test_data_path}")
    print(f"Train rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Features: {MODEL_FEATURES}")
    print("Feature mode: ratio_features")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")

    if output_model_path:
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        with output_model_path.open("wb") as model_file:
            pickle.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "features": MODEL_FEATURES,
                    "target": TARGET,
                    "feature_mode": "ratio_features",
                    "train_data": str(train_data_path),
                    "test_data": str(test_data_path),
                },
                model_file,
            )
        print(f"Saved model to {output_model_path}")

    if predictions_path:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        output = x_test.copy()
        output["actual_ipc"] = y_test.to_numpy()
        output["predicted_ipc"] = predictions
        output["residual"] = output["actual_ipc"] - output["predicted_ipc"]
        output.to_csv(predictions_path, index=False)
        print(f"Saved predictions to {predictions_path}")

        summary = pd.DataFrame(
            {
                "Model": ["Ratio-Feature MLP"],
                "RMSE": [rmse],
                "MAE": [mae],
                "R2": [r2],
            }
        )
        summary_path = predictions_path.parent / "mlp_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_train_data_path = project_root / "training_data.csv"
    default_test_data_path = project_root / "testing_data.csv"

    parser = argparse.ArgumentParser(
        description="Train an MLP regressor for IPC prediction using ratio features."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=default_train_data_path,
        help="CSV file containing the training split.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=default_test_data_path,
        help="CSV file containing the testing split.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=project_root / "mlp_results" / "mlp_ipc_model.pkl",
        help="Where to save the trained model bundle. Use an empty string to skip saving.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project_root / "mlp_results" / "mlp_regression_predictions.csv",
        help="Where to save test-set predictions. Use an empty string to skip saving.",
    )
    return parser.parse_args()


def optional_path(path):
    if path is None or str(path).strip() == "":
        return None
    return path


if __name__ == "__main__":
    args = parse_args()
    train_mlp(
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        output_model_path=optional_path(args.output_model),
        predictions_path=optional_path(args.predictions),
    )

import os
import json
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DATA_DIR = "data/processed_data"
RAW_DIR = "data"
MODELS_DIR = "models"
METRICS_DIR = "metrics"

MODEL_PATH = os.path.join(MODELS_DIR, "elasticnet_model.pkl")
PRED_PATH = os.path.join(RAW_DIR, "predictions.csv")
SCORES_PATH = os.path.join(METRICS_DIR, "scores.json")

def main():
    os.makedirs(METRICS_DIR, exist_ok=True)

    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    metrics = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    with open(SCORES_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Sauver un dataset avec y_test et prediction
    out = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    out.to_csv(PRED_PATH, index=False)

if __name__ == "__main__":
    main()
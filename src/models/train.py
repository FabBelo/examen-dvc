import os
import pandas as pd
import joblib

from sklearn.linear_model import ElasticNet

DATA_DIR = "data/processed_data"
MODELS_DIR = "models"

BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_params.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "elasticnet_model.pkl")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

    best_params = joblib.load(BEST_PARAMS_PATH)

    elasticnet_model = ElasticNet(max_iter=10000, **best_params)
    elasticnet_model.fit(X_train, y_train)

    joblib.dump(elasticnet_model, MODEL_PATH)

if __name__ == "__main__":
    main()
import os
import pandas as pd
import joblib

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

DATA_DIR = "data/processed_data"
OUT_DIR = "models"
BEST_PARAMS_PATH = os.path.join(OUT_DIR, "best_params.pkl")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

    model = ElasticNet(max_iter=10000)

    param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    joblib.dump(gs.best_params_, BEST_PARAMS_PATH)

if __name__ == "__main__":
    main()
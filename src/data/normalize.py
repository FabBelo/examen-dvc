import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

IN_DIR = "examen-dvc/data/processed_data"
OUT_DIR = "examen-dvc/data/processed_data"
SCALER_PATH = "examen-dvc/models/scaler.pkl"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("examen-dvc/models", exist_ok=True)

    X_train = pd.read_csv(os.path.join(IN_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(IN_DIR, "X_test.csv"))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled.to_csv(os.path.join(OUT_DIR, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(OUT_DIR, "X_test_scaled.csv"), index=False)

    joblib.dump(scaler, SCALER_PATH)

if __name__ == "__main__":
    main()
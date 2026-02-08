import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw_data/raw.csv"
OUT_DIR = "data/processed_data"
TARGET_COL = "silica_concentrate"

def main(test_size: float = 0.2, random_state: int = 42):
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    df = df.select_dtypes(include=["number"])
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

if __name__ == "__main__":
    main()
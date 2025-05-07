# train_icu_stay.py - Predict ICU length of stay from early vital signs and demographics

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# === Configuration ===
INPUT_FILE = "data/processed/icu_features.parquet"


def load_data():
    df = pd.read_parquet(INPUT_FILE)
    if "los_hours" not in df.columns:
        raise ValueError("Missing 'los_hours' column. Check cleaning.py to compute it.")
    return df


def preprocess(df):
    # Drop irrelevant columns
    drop_cols = [
        "icustay_id",
        "subject_id",
        "intime",
        "outtime",
        "gender",
        "hospital_expire_flag",
    ]
    df = df.drop(
        columns=[col for col in drop_cols if col in df.columns], errors="ignor"
    )

    # Features and target
    X = df.drop(columns=["los_hours"])
    y = df["los_hours"]

    # Handle missing values
    X = X.fillna(X.mean(numeric_only=True))
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Evaluation Results")
    print("----------------------")
    print("Mean Absolute Error :", mean_absolute_error(y_test, y_pred))
    print("R^2 Score           :", r2_score(y_test, y_pred))

    return model


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    X, y = preprocess(df)

    print("Training model...")
    model = train_model(X, y)

# train_sepsis_risk.py - Predict Sepsis/Shock/Respiratory Failure risk using ICU stay features

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

INPUT_FILE = "data/processed/icu_features.parquet"


def load_data():
    df = pd.read_parquet(INPUT_FILE)
    if "sepsis_shock_respfail_flag" not in df.columns:
        raise ValueError(
            "Missing 'sepsis_shock_respfail_flag' column. Check cleaning.py to merge it."
        )
    return df


def preprocess(df):
    drop_cols = [
        "icustay_id",
        "subject_id",
        "intime",
        "outtime",
        "gender",
        "hospital_expire_flag",
        "los_hours",
    ]
    df = df.drop(
        columns=[col for col in drop_cols if col in df.columns], errors="ignore"
    )
    X = df.drop(columns=["sepsis_shock_respfail_flag"])
    y = df["sepsis_shock_respfail_flag"]
    X = X.fillna(X.mean(numeric_only=True))
    return X, y


def train_model(X, y):
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, stratify=stratify, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    return model


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    X, y = preprocess(df)

    print("Training model...")
    model = train_model(X, y)

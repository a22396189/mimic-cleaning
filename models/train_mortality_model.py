# Train a predictive model for ICU mortality using cleaned MIMIC-IV data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os

# === Configuration ===
INPUT_FILE = "data/processed/icu_features.parquet"


def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    return df


def preprocess(df):
    # Drop columns not used for modeling
    drop_cols = ["icustay_id", "subject_id", "intime", "outtime", "gender"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    if "hospital_expire_flag" not in df.columns:
        raise ValueError(
            "Target column 'hospital_expire_flag' not found. Did you merge it?"
        )

    # Separate features and target
    X = df.drop(columns=["hospital_expire_flag"])
    y = df["hospital_expire_flag"]

    # Fill missing values
    X = X.fillna(X.mean(numeric_only=True))
    return X, y


def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, digits=3)
    auc = roc_auc_score(y_test, y_prob)

    print("=== Classification Report ===")
    print(report)
    print(f"ROC AUC Score: {auc:.3f}")

    return clf


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    X, y = preprocess(df)

    print("Training model...")
    model = train_model(X, y)

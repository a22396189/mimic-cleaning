# train_readmission_risk.py - Predict ICU Readmission Risk

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# === CONFIG ===
DATA_PATH = "data/processed/icu_features.csv"
TARGET_COL = "readmission_flag"

# === LOAD ===
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# === PREPROCESSING ===
print("Preprocessing...")
df = df.dropna(subset=[TARGET_COL])
df = df.drop(columns=["icustay_id", "subject_id", "intime", "outtime"])

X = df.drop(columns=[TARGET_COL])
X = pd.get_dummies(X)  # encode gender
y = df[TARGET_COL]

# === TRAINING ===
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === EVALUATION ===
print("=== Classification Report ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

try:
    y_prob = model.predict_proba(X_test)[:, 1]
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
except Exception:
    print("ROC AUC could not be computed (likely due to 1 class only).")

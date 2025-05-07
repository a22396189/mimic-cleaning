# cleaning.py - Script to process raw MIMIC-IV data into a clean, analysis-ready ICU stay-level dataset

import os
from datetime import timedelta

import pandas as pd

from feature_map import VITALS_ITEMIDS

# === CONFIGURATION ===
RAW_DATA_DIR = "data/raw"
OUTPUT_FILE = "data/processed/icu_features.parquet"
TIME_WINDOW_HOURS = 24


# === DATA LOADING ===
def load_csv(file_name):
    return pd.read_csv(os.path.join(RAW_DATA_DIR, file_name))


def load_all_tables():
    patients = load_csv("patients.csv")
    icustays = load_csv("icustays.csv")
    chartevents = load_csv("chartevents.csv")
    d_items = load_csv("d_items.csv")  # optional
    return patients, icustays, chartevents, d_items


# === STATIC FEATURE EXTRACTION ===
def extract_static_features(patients, icustays):
    df = icustays.merge(patients, on="subject_id", how="left")
    df["age"] = df["anchor_age"]
    return df[["icustay_id", "subject_id", "age", "gender", "intime", "outtime"]]


# === VITALS FEATURE EXTRACTION ===
def extract_vitals(chartevents, base):
    vitals_df = chartevents[chartevents["itemid"].isin(VITALS_ITEMIDS.keys())].copy()
    vitals_df["charttime"] = pd.to_datetime(vitals_df["charttime"])
    vitals_df["valuenum"] = pd.to_numeric(vitals_df["valuenum"], errors="coerce")

    feature_rows = []
    for _, row in base.iterrows():
        stay_id = row["icustay_id"]
        intime = pd.to_datetime(row["intime"])

        stay_data = vitals_df[
            (vitals_df["subject_id"] == row["subject_id"])
            & (vitals_df["charttime"] >= intime)
            & (vitals_df["charttime"] < intime + timedelta(hours=TIME_WINDOW_HOURS))
        ]

        summary = {"icustay_id": stay_id}
        for itemid, name in VITALS_ITEMIDS.items():
            vals = stay_data[stay_data["itemid"] == itemid]["valuenum"].dropna()
            summary[f"{name}_mean"] = vals.mean() if not vals.empty else None
            summary[f"{name}_max"] = vals.max() if not vals.empty else None

        feature_rows.append(summary)

    return pd.DataFrame(feature_rows)


# === MAIN AGGREGATION ===
def aggregate_features():
    print("Loading data...")
    patients, icustays, chartevents, d_items = load_all_tables()

    print("Extracting static features...")
    static_features = extract_static_features(patients, icustays)

    print("Calculating length of stay (LOS)...")
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    icustays["los_hours"] = (
        icustays["outtime"] - icustays["intime"]
    ).dt.total_seconds() / 3600
    static_features = static_features.merge(
        icustays[["icustay_id", "los_hours"]], on="icustay_id", how="left"
    )

    print("Aggregating vital signs...")
    vitals = extract_vitals(chartevents, static_features)

    print("Merging features...")
    features = static_features.merge(vitals, on="icustay_id", how="left")

    print("Merging mortality labels from admissions.csv...")
    admissions = load_csv("admissions.csv")
    features = features.merge(
        admissions[["subject_id", "hospital_expire_flag"]], on="subject_id", how="left"
    )

    print("Merging sepsis/shock/respfail labels from sepsis_labels.csv...")
    sepsis = load_csv("sepsis_labels.csv")
    features = features.merge(
        sepsis[["subject_id", "icustay_id", "sepsis_shock_respfail_flag"]],
        on=["subject_id", "icustay_id"],
        how="left",
    )
    features["sepsis_shock_respfail_flag"] = (
        features["sepsis_shock_respfail_flag"].fillna(0).astype(int)
    )

    print("Merging readmission labels from readmission_labels.csv...")
    readmit = load_csv("readmission_labels.csv")
    features = features.merge(
        readmit[["subject_id", "icustay_id", "readmission_flag"]],
        on=["subject_id", "icustay_id"],
        how="left",
    )
    features["readmission_flag"] = features["readmission_flag"].fillna(0).astype(int)

    if "hospital_expire_flag" not in features.columns:
        raise ValueError("Missing hospital_expire_flag after merge!")

    print("Saving cleaned dataset...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    features.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to: {OUTPUT_FILE}")

    csv_output_path = OUTPUT_FILE.replace(".parquet", ".csv")
    features.to_csv(csv_output_path, index=False)
    print(f"Also saved to: {csv_output_path}")
    print("Final shape:", features.shape)


# === ENTRY POINT ===
if __name__ == "__main__":
    aggregate_features()

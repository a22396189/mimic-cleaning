# cleaning.py - Script to process raw MIMIC-IV data into a clean, analysis-ready ICU stay-level dataset

import pandas as pd
import os
from datetime import timedelta
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
    d_items = load_csv("d_items.csv")  # optional, not used yet
    return patients, icustays, chartevents, d_items


# === STATIC FEATURE EXTRACTION ===
def extract_static_features(patients, icustays):
    """
    Merge patient demographics with ICU stay info, and extract age, gender, intime/outtime.
    """
    df = icustays.merge(patients, on="subject_id", how="left")
    df["age"] = df[
        "anchor_age"
    ]  # or compute from anchor_year and anchor_year_group if needed
    return df[["icustay_id", "subject_id", "age", "gender", "intime", "outtime"]]


# === VITALS FEATURE EXTRACTION ===
def extract_vitals(chartevents, base):
    """
    Extract time-series vital sign features for each ICU stay:
    - Aggregates: mean and max values over first 24 hours
    """
    # 1. Filter only relevant itemids
    vitals_df = chartevents[chartevents["itemid"].isin(VITALS_ITEMIDS.keys())].copy()
    vitals_df["charttime"] = pd.to_datetime(vitals_df["charttime"])
    vitals_df["valuenum"] = pd.to_numeric(vitals_df["valuenum"], errors="coerce")

    # 2. Prepare output
    feature_rows = []

    for _, row in base.iterrows():
        stay_id = row["icustay_id"]
        intime = pd.to_datetime(row["intime"])

        # 3. Filter 24hr time window for this stay
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
    base = extract_static_features(patients, icustays)

    print("Aggregating vital signs...")
    vitals = extract_vitals(chartevents, base)

    print("Merging features...")
    features = base.merge(vitals, on="icustay_id", how="left")

    print("Saving cleaned dataset...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    features.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Final shape: {features.shape}")


# === ENTRY POINT ===
if __name__ == "__main__":
    aggregate_features()

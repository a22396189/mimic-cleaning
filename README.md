# MIMIC-IV: Data Cleaning and Transformation Pipeline

This project focuses on cleaning and transforming raw MIMIC-IV tables into an analysis-friendly format for downstream machine learning and statistical analysis. It aggregates vital signs, lab results, and demographic features into a wide-format table at the ICU stay level (`icustay_id`).

---

## Objectives

- Extract key features from `chartevents`, `labevents`, and patient tables.
- Aggregate time-series data (first 24hr of ICU stay).
- Produce a single, clean DataFrame with one row per `icustay_id`.
- Store data in efficient formats for reuse in modeling tasks.

---

## Data Sources

```
| Table | Description |
|-------|-------------|
| `patients.csv` | Basic demographic info |
| `admissions.csv` | Hospital admission data |
| `icustays.csv` | ICU-specific episodes |
| `chartevents.csv` | Vital signs (high-frequency) |
| `labevents.csv` | Lab results |
| `d_items.csv` | Mapping `itemid` to label |
```

---

## Project Structure

```
mimic-cleaning/
├── data/
│ ├── raw/ # Raw CSV files
│ └── processed/ # Output cleaned data
├── notebooks/ # Optional: EDA and analysis
├── src/
│ ├── cleaning.py # Main data cleaning logic
│ ├── feature_map.py # Custom itemid to feature mapping
├── requirements.txt
└── README.md
```

---

## How to Run

1. Download MIMIC-IV CSVs and place under `data/raw/`.
2. Create conda environment:
   ```bash
   conda create -n mimic-clean python=3.10 pandas numpy pyarrow
   conda activate mimic-clean
   pip install -r requirements.txt
3. Run the cleaning script:
   ```bash
   python src/cleaning.py

---

## Output Format: `icu_features.parquet`

The output file `icu_features.parquet` is stored in **[Apache Parquet](https://parquet.apache.org/)** format — a compressed, efficient, and columnar storage format ideal for large-scale data analysis.

You can read it using Python with `pandas`:

```python
import pandas as pd

df = pd.read_parquet("data/processed/icu_features.parquet")
print(df.head())
```

If needed, you can also convert it to CSV for easier inspection:
```python
df.to_csv("data/processed/icu_features.csv", index=False)
```

---
## License
This project is released under the MIT License.
You are free to use, modify, and distribute this code with proper attribution.

---

## Acknowledgements

This project uses data derived from the **MIMIC-IV v2.2** database, a freely accessible critical care dataset developed and maintained by the **MIT Laboratory for Computational Physiology**.

Access to MIMIC-IV requires credentialing and completion of the appropriate data use agreements through PhysioNet.

- MIMIC-IV v2.2 official page: https://physionet.org/content/mimiciv/2.2/
- Citation: Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67.

We gratefully acknowledge the researchers and contributors who make MIMIC-IV available to the public.

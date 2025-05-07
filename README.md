# MIMIC-IV: Data Cleaning and Predictive Modeling Pipeline

This project provides a pipeline for cleaning raw MIMIC-IV data and building predictive models to support critical care decisions in the ICU. The pipeline includes feature extraction and machine learning models for mortality prediction, length of stay estimation, complication risk assessment, and ICU readmission prediction.

---

## Objectives

- Clean and transform raw MIMIC-IV CSV files into an analysis-friendly format
- Aggregate static and dynamic ICU features over the first 24 hours
- Enable downstream modeling by exporting clean datasets in `.parquet` and `.csv` format
- Implement 4 key predictive modeling tasks

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
All data files are assumed to be in `data/raw/`. This project uses MIMIC-IV v2.2 from [PhysioNet](https://physionet.org/content/mimiciv/2.2/).

---

## Project Structure

```
mimic-cleaning/
├── data/
│ ├── raw/ # Raw CSVs (excluded from Git)
│ └── processed/ # Output features (ignored by Git)
├── models/ # Modeling scripts for each task
│ ├── train_mortality_model.py
│ ├── train_icu_stay.py
│ ├── train_sepsis_risk.py
│ └── train_readmission_risk.py
├── notebooks/ # Optional: EDA or analysis notebooks
├── src/
│ ├── cleaning.py # Main feature extraction logic
│ └── feature_map.py # Custom ITEMID-to-feature mapping
├── LICENSE
├── requirements.txt
├── .gitignore
└── README.md
```

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

## Predictive Models

1. ICU or Hospital Mortality Prediction
```bash
python models/train_mortality_model.py
```

2. Length of Stay (LOS) Prediction
```bash
python models/train_icu_stay.py
```

3. Sepsis / Shock / Respiratory Failure Risk
```bash
python models/train_sepsis_risk.py
```

4. ICU Readmission Risk
```bash
python models/train_readmission_risk.py
```

---

## Setup Instructions

1. Download MIMIC-IV CSVs and place under `data/raw/`.
2. Create conda environment:
   ```bash
   conda create -n mimic-clean python=3.10 pandas numpy pyarrow
   conda activate mimic-clean
   pip install -r requirements.txt
3. Run the cleaning script (extract features):
   ```bash
   python src/cleaning.py

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

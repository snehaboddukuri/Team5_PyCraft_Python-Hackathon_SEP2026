# Hospitalized Heart Failure Patients Analytics — Team05 PyCraft

**Python Hackathon · September 2026**
Team: **Snehalatha · Rashmi · Sandhya · Jayasree**

An end-to-end analysis of 2,008 hospitalized heart-failure patients: data cleaning, descriptive, prescriptive and predictive (machine learning) analysis, and an interactive Streamlit dashboard.

---

## 📊 Dataset

| | |
|---|---|
| **Title** | Hospitalized patients with heart failure: integrating electronic healthcare records and external outcome data |
| **Source** | PhysioNet — Zhang Z. et al., *Scientific Data* 8, 46 (2021) |
| **Hospital** | Zigong Fourth People's Hospital, Sichuan, China (2016–2019) |
| **Size** | 2,008 patients, 166 attributes, split across 7 CSV files |
| **Outcomes** | Death and readmission at 28 days, 3 months and 6 months |

---

## 📁 Repository structure

```
Team5_PyCraft_Python-Hackathon_SEP2026/
├── cardiac_failure/                      # Raw data (7 CSV files)
├── Team05_PyCraft_Cleaned_data/          # Cleaned data (7 cleaned CSVs + wide_merged_table.csv)
├── Team05_PyCraft_1.Cleaning.ipynb       # Category 1 – Data cleaning
├── Team05_PyCraft_2.Descriptive.ipynb    # Category 2 – Descriptive analysis
├── Team05_PyCraft_3.Prescriptive.ipynb   # Category 3 – Prescriptive analysis
├── Team05_PyCraft_4.Predictive.ipynb     # Category 4 – Predictive analysis (ML)
├── Team05_PyCraft_5.Dashboard.py         # Streamlit dashboard
└── README.md
```

---

## 📓 Project files

Run the notebooks **in this order** — each one uses the output of the one before.

| # | File | Category | What it does | Input |
|---|---|---|---|---|
| 1 | [Team05_PyCraft_1.Cleaning.ipynb](Team05_PyCraft_1.Cleaning.ipynb) | Data Cleaning | Cleans each of the 7 raw tables: column renaming, text standardization, data types, duplicates, clinical outliers, missing values. Saves 7 cleaned CSVs. | `cardiac_failure/*.csv` |
| 2 | [Team05_PyCraft_2.Descriptive.ipynb](Team05_PyCraft_2.Descriptive.ipynb) | Descriptive | 10 questions — demographics, outcomes, HF type, severity, consciousness, medicines, lab markers, comorbidities. | 7 cleaned CSVs |
| 3 | [Team05_PyCraft_3.Prescriptive.ipynb](Team05_PyCraft_3.Prescriptive.ipynb) | Prescriptive | Merges the 7 cleaned tables into `wide_merged_table.csv`, then answers 30 questions, each with key insights and a recommended action. | 7 cleaned CSVs |
| 4 | [Team05_PyCraft_4.Predictive.ipynb](Team05_PyCraft_4.Predictive.ipynb) | Predictive | 10 machine-learning questions (Logistic Regression, Random Forest, 5-fold cross-validation) with accuracy, precision, recall, F1, ROC-AUC, confusion matrix and classification report. | `wide_merged_table.csv` |
| 5 | [Team05_PyCraft_5.Dashboard.py](Team05_PyCraft_5.Dashboard.py) | Dashboard | Interactive Streamlit dashboard with KPIs and three tabs. | `wide_merged_table.csv` |

---

## 🗂️ Source files

### Raw files — `cardiac_failure/`

| File | Contents |
|---|---|
| `demography.csv` | Gender, weight, height, BMI, occupation, age category |
| `hospitalization_discharge.csv` | Admission & discharge details, length of stay, hospital outcome, death/readmission at 28 days, 3 and 6 months |
| `cardiac_complications.csv` | NYHA class, Killip grade, heart-failure type, LVEF and echo measurements |
| `patienthistory.csv` | Comorbidities (diabetes, chronic kidney disease, COPD, dementia, cancer, …) |
| `labs.csv` | Admission vital signs and ~100 lab tests (kidney function, potassium, troponin, BNP, …) |
| `patient_precriptions.csv` | Medicines prescribed (one row per patient per drug) |
| `responsivenes.csv` | Consciousness level and Glasgow Coma Scale (eye, verbal, motor) |

### Cleaned files — `Team05_PyCraft_Cleaned_data/`

| File | Created by | Used by |
|---|---|---|
| `demography_cleaned.csv`, `hospitalization_discharge_cleaned.csv`, `cardiac_cleaned.csv`, `patienthistory_cleaned.csv`, `labs_cleaned.csv`, `precriptions_cleaned.csv`, `responsivenes_cleaned.csv` | 1.Cleaning | 2.Descriptive, 3.Prescriptive |
| **`wide_merged_table.csv`** — one row per patient (2,008 rows) combining all 7 tables | 3.Prescriptive | 3.Prescriptive, 4.Predictive, 5.Dashboard |

---

## 🖥️ Dashboard

The dashboard has **8 KPI cards** and three tabs, plus sidebar filters (gender, age group, HF type, admission type, NYHA class).

| Tab | Question | Highlights |
|---|---|---|
| **Descriptive** | What happened? | Age & gender, malnutrition, readmission/mortality over time, severity, consciousness curves, medicines, comorbidities |
| **Prescriptive** | What should we do? | Kidney (eGFR) stages, follow-up heatmap, cardio-renal & potassium danger zones, warning signs, comorbidity risk, care-process gaps — each with an action box |
| **Predictive** | What will happen? | Model leaderboard, interactive model explorer (ROC curve, confusion matrix, metrics, feature importance, risk deciles) and a bedside risk calculator |

### How to run

**1. Clone the repository** (or download it as a ZIP from GitHub)
```bash
git clone https://github.com/snehaboddukuri/Team5_PyCraft_Python-Hackathon_SEP2026.git
cd Team5_PyCraft_Python-Hackathon_SEP2026
```

**2. Install the required packages** (first time only)
```bash
pip install streamlit pandas numpy plotly matplotlib scikit-learn
```

**3. Start the dashboard**
```bash
streamlit run Team05_PyCraft_5.Dashboard.py
```

**4.** The dashboard opens in your browser at **http://localhost:8501**. Press **Ctrl + C** in the terminal to stop it.

> **Tips**
> - If you get `'streamlit' is not recognized`, run: `python -m streamlit run Team05_PyCraft_5.Dashboard.py`
> - Keep `Team05_PyCraft_Cleaned_data/wide_merged_table.csv` in place — the dashboard reads it automatically (or upload it from the sidebar).
> - The first time a model is opened in the Predictive tab it trains for about 10–40 seconds; after that it is cached.

### Running the notebooks
```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn statsmodels
jupyter notebook
```
Open the notebooks from the project folder and run them in order (1 → 4).

---

## 💡 Key findings

| | Finding |
|---|---|
| **KPIs** | 38.5% of patients are readmitted within 6 months, but only 2.8% die — readmission is about 14× more common. |
| **Descriptive** | An elderly, frail cohort: 73% are aged 69+ and 1 in 4 is underweight. Readmission builds for 6 months; deaths cluster in the first weeks. |
| **Prescriptive** | Moderate kidney disease drives readmission (50%) while severe kidney disease drives death; potassium > 5.0 triples mortality; patients with no echo (50% vs 33% readmitted) or who leave against advice (18.7% 28-day death) need process fixes. |
| **Predictive** | Early death is predictable from admission data (ROC-AUC **0.90** at 28 days). Readmission is not (AUC ≤ 0.67) — it depends on what happens after discharge. |

---

## 🛠️ Tech stack

Python · pandas · NumPy · Matplotlib · Seaborn · Plotly · scikit-learn · statsmodels · Streamlit · Jupyter

---

## 👥 Team05 PyCraft

Snehalatha · Rashmi · Sandhya · Jayasree


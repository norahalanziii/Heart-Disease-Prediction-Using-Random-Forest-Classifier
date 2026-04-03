# 🫀 Heart Disease Prediction Using Random Forest Classifier

---

---

## 📌 Project Overview

This project develops a machine learning model for early heart disease prediction using data mining techniques. The model is built on a dataset of 2,182 patient records containing 14 medical and demographic features. Four classification algorithms were implemented and compared, with the **Random Forest Classifier** achieving the best performance at **96.65% accuracy** after hyperparameter tuning.

The goal is to support healthcare providers — particularly in Saudi Arabia — in identifying high-risk patients early, enabling timely intervention and reducing mortality rates.

---

## 📁 Project Files

```
├── Data_mining_Final_CODE.py          # Full pipeline: cleaning → EDA → modeling
├── Uncleaned_Version.csv              # Raw dataset with injected noise
├── cleaned_data_v2_after_chang.xlsx   # Final cleaned dataset
├── Data_Mining_Report.pdf             # Full project report
└── README.md                          # This file
```

---

## 📊 Dataset

- **Source:** [Kaggle – Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset/)
- **Records:** 2,182 (merged from 5 public heart disease datasets)
- **Features:** 14 (13 input features + 1 target)

### Feature Description

| Feature | Description |
|---|---|
| `age` | Patient age (numeric) |
| `sex` | Gender: 1 = Male, 0 = Female |
| `cp` | Chest pain type: 0–3 |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl: 1 = True, 0 = False |
| `restecg` | Resting ECG results: 0–2 |
| `thalachh` | Max heart rate achieved |
| `exang` | Exercise-induced angina: 1 = Yes, 0 = No |
| `oldpeak` | ST depression (exercise vs. rest) |
| `slope` | Slope of peak exercise ST segment: 0–2 |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia: 1 = Normal, 2 = Fixed defect, 3 = Reversible defect |
| `target` | **Heart attack risk: 1 = High, 0 = Low** |

---

## ⚙️ Pipeline

### 1. Data Preparation & Cleaning
- Removed **267 duplicate rows**
- Imputed missing values using **mean/median** (numeric) and **mode** (categorical)
- Standardized inconsistent string labels (e.g., `"male"`, `"m"` → `"Male"`)
- Converted string-stored numeric columns to proper types
- Capped outliers using **IQR + percentile method**
- Converted measurement units (`trestbps`: kPa → mmHg, `chol`: mmol/L → mg/dL)

### 2. Data Transformation
- `sex` column: `"Male"/"Female"` → `1/0`
- `fbs` column: `"True"/"False"` → `1/0`
- All columns normalized to consistent formats and types

### 3. Data Selection
- Dropped the `country` column (irrelevant to the study objectives)
- Retained all 13 medically relevant features

### 4. Data Splitting
- **70% Training** (1,321 records) / **30% Testing** (567 records)
- `random_state=42` for reproducibility

### 5. Exploratory Data Analysis (EDA)

Key statistics from the training set:

| Attribute | Mean | Median | Variance |
|---|---|---|---|
| Age | 53.40 | 54.0 | 73.20 |
| Cholesterol | 246.37 | 240.0 | 2883.14 |
| Resting BP | 131.80 | 130.0 | 307.61 |
| Max Heart Rate | 147.94 | 150.0 | 473.36 |

Notable findings:
- Cholesterol and resting blood pressure are **right-skewed** with significant outliers
- Age shows an **inverse correlation** with maximum heart rate
- Cholesterol levels show a **weak positive correlation** with age and resting blood pressure

### 6. Modeling

Four classifiers were trained and evaluated:

| Model | Accuracy (Before Tuning) | Accuracy (After Tuning) |
|---|---|---|
| **Random Forest** | 96.12% | **96.65%** |
| Decision Tree | 95.59% | 94.18% |
| Logistic Regression | 73.19% | 73.90% |
| KNN | 74.07% | 83.95% |

Validation: **5-Fold Cross-Validation** + **GridSearchCV** for hyperparameter tuning.

Best parameters for Random Forest: `max_depth=10`, `min_samples_split=2`, `n_estimators=200`

---

## 🔑 Key Risk Indicators Identified

The model identified the following as the most significant predictors of heart disease:
- Elevated **cholesterol levels** (`chol`)
- Increased **resting blood pressure** (`trestbps`)
- **Advanced age**
- **Lower maximum heart rate** (`thalachh`)

---

## 🚀 How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Run the Pipeline

```bash
python Data_mining_Final_CODE.py
```

> Make sure `uncleaned_data_v2.csv` is in the same directory. If not found, the script will fall back to `cleaned_data_v2_after_chang.xlsx`.

### Output
- Cleaned dataset: `cleaned_data_v2.csv`
- Visualizations saved to: `visualizations/` folder
- Model performance printed to console

---

## 📈 Results Summary

The tuned **Random Forest Classifier** is the recommended model:

| Metric | Score |
|---|---|
| Accuracy | 96.65% |
| Precision | 96.65% |
| Recall | 96.65% |
| F1-Score | 96.65% |

---

## ⚠️ Limitations

- Dataset covers only patients **aged 38 and older** — may not generalize to younger individuals
- Trained on a single dataset, which may limit generalizability across diverse populations

---

## 🔭 Future Work

- Incorporate diverse, multi-hospital datasets
- Expand age range coverage to include younger patients
- Explore additional algorithms to further reduce the **False Negative (FN)** rate

---

## 📚 References

1. Ministry of Health, Saudi Arabia – *Heart diseases are the cause of 42% of non-communicable disease deaths in the Kingdom*
2. GeeksforGeeks – [What is Kaggle?](https://www.geeksforgeeks.org/machine-learning/what-is-kaggle/)
3. Kaggle – [kaggle.com](https://www.kaggle.com/)
4. J. Han, J. Pei, and H. Tong – *Data Mining*, Morgan Kaufmann, 2022
5. Farhaan Nazirkhan – [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset/)

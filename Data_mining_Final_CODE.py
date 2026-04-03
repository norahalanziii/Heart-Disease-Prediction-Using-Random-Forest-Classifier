import pandas as pd
import numpy as np
import os
import math
from collections import Counter
import io
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")


# ---------------- CONFIG ----------------
INPUT_CSV = "uncleaned_data_v2.csv"
OUTPUT_CSV = "cleaned_data_v2.csv"
DROP_COL_THRESHOLD = 0.60
OUTLIER_CAPPING = True
OUTLIER_CAP_LOWER = 0.05
OUTLIER_CAP_UPPER = 0.95
RANDOM_STATE = 42
# ----------------------------------------

# ------------------------------------------------------- Data Preparation Phase--------------------------------------------------
# This phase includes Data Cleaning and Data Transformation.
def report_basic_info(df, name="data"):
    print(f"\n---- Basic report for {name} ----")
    print("Shape:", df.shape)
    print("Columns and dtypes:")
    print(df.dtypes)
    print("\nMissing values (count):")
    print(df.isna().sum())
    print("\nSample unique values (up to 10) for object/categorical columns:")
    for c in df.select_dtypes(include=['object']).columns:
        try:
            vals = df[c].dropna().unique()[:10]
            print(f" - {c}: {vals}")
        except Exception:
            pass
    print("---- end report ----\n")

def coerce_numeric_if_possible(series):
    # Try to coerce strings like ' 12 ' or '12.0' or '1,234' to numeric
    if series.dtype == 'O':
        # Remove commas, trim whitespace
        cleaned = series.str.replace(',', '').str.strip()
        coerced = pd.to_numeric(cleaned, errors='coerce')
        # if a large fraction converts to numbers, keep it
        converted_frac = coerced.notna().mean()
        if converted_frac >= 0.6:
            return coerced
        else:
            return series
    else:
        return series

def standardize_sex(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ('m', 'male', 'man', 'boy', '1'):
        return 'Male'
    if s in ('f', 'female', 'woman', 'girl', '0'):
        return 'Female'
    # keep original if unknown
    return val

def standardize_boolean_like(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ('true', 't', 'yes', 'y', '1'):
        return 1
    if s in ('false', 'f', 'no', 'n', '0'):
        return 0
    # if already numeric 1/0, return as is
    try:
        if float(s) == 1.0:
            return 1
        if float(s) == 0.0:
            return 0
    except Exception:
        pass
    return val

def impute_numeric(series):
    # choose mean or median depending on skew
    if series.dropna().shape[0] == 0:
        return series
    skew = series.dropna().skew()
    if abs(skew) > 1:  # strongly skewed -> median
        fill = series.median()
    else:
        fill = series.mean()
    return series.fillna(fill)

def impute_categorical(series):
    if series.dropna().shape[0] == 0:
        return series
    mode = series.mode()
    if len(mode) == 0:
        return series.fillna('Unknown')
    return series.fillna(mode.iloc[0])

def cap_outliers_iqr(series):
    if series.dtype.kind not in 'biufc':  # not numeric
        return series
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return series
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    p_low = series.quantile(OUTLIER_CAP_LOWER)
    p_high = series.quantile(OUTLIER_CAP_UPPER)
    lower_cap = max(lower, p_low)
    upper_cap = min(upper, p_high)
    return series.clip(lower=lower_cap, upper=upper_cap)

# --- Data Preparation Execution (Data Cleaning) ---

if not os.path.exists(INPUT_CSV):
    # Try to load from the file path used in the second snippet if INPUT_CSV is not found
    print(f"{INPUT_CSV} not found. Attempting to load from the path in the second code block.")
    file_path = "cleaned_data_v2_after chang(1).xlsx"
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        raise FileNotFoundError(f"Neither {INPUT_CSV} nor {file_path} found in working directory.")
else:
    df = pd.read_csv(INPUT_CSV)
    print("Loaded file:", INPUT_CSV)
    report_basic_info(df, "raw")

    # Drop columns with too many missing values
    missing_frac = df.isna().mean()
    cols_to_drop = missing_frac[missing_frac > DROP_COL_THRESHOLD].index.tolist()
    if cols_to_drop:
        print("Dropping columns with >{:.0%} missing: {}".format(DROP_COL_THRESHOLD, cols_to_drop))
        df = df.drop(columns=cols_to_drop)
    else:
        print("No columns exceed missingness threshold.")

    # Trim whitespace & normalize string columns
    for c in df.select_dtypes(include=['object']).columns:
        # strip whitespace
        df[c] = df[c].astype(str).str.strip()
        # replace common null strings with actual NaN
        df[c] = df[c].replace({'nan': np.nan, 'None': np.nan, 'none': np.nan, 'NA': np.nan, 'N/A': np.nan, '': np.nan})

    # Convert obvious numeric columns stored as strings
    for c in df.columns:
        coerced = coerce_numeric_if_possible(df[c])
        if coerced is not df[c]:
            print(f"Coerced column {c} to numeric (string->numeric).")
            df[c] = coerced

    # Standardize expected categorical columns (example: sex, gender, fbs)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].apply(standardize_sex)
    # Note: 'fbs' is left as is for now, mapping will happen in Transformation

    # Remove exact duplicates
    n_before = df.shape[0]
    df = df.drop_duplicates(ignore_index=True)
    n_after = df.shape[0]
    print(f"Removed {n_before - n_after} exact duplicate rows.")

    # Impute missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert numeric dtypes to floats for imputation
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Impute numeric
    for c in numeric_cols:
        if df[c].isna().any():
            before_na = df[c].isna().sum()
            df[c] = impute_numeric(df[c])
            after_na = df[c].isna().sum()
            print(f"Numeric impute: column '{c}': filled {before_na - after_na} missing values using mean/median.")

    # Impute categorical
    for c in object_cols:
        if df[c].isna().any():
            before_na = df[c].isna().sum()
            df[c] = impute_categorical(df[c])
            after_na = df[c].isna().sum()
            print(f"Categorical impute: column '{c}': filled {before_na - after_na} missing values with mode.")

    # Outlier handling (IQR capping)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts = {}
    for c in numeric_cols:
        series = df[c].dropna()
        if series.shape[0] < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[c] < lower) | (df[c] > upper)).sum()
        if n_outliers > 0:
            outlier_counts[c] = int(n_outliers)
            if OUTLIER_CAPPING:
                df[c] = cap_outliers_iqr(df[c])
    if outlier_counts:
        print("Outliers detected (column: count):", outlier_counts)
        if OUTLIER_CAPPING:
            print("Outliers have been capped using IQR + percentile heuristics.")
    else:
        print("No prominent outliers detected by IQR method.")

    # normalize some categorical names (title case)
    for c in df.select_dtypes(include=['object']).columns:
        # convert strings like 'male'/'Male'-> 'Male'
        df[c] = df[c].apply(lambda x: x.title() if isinstance(x, str) and len(x) < 50 else x)


# --- Data Preparation Execution (Data Transformation) ---

print("\n--- Starting Data Transformation ---")

# Ensure numeric columns are numeric before transformation
if "trestbps" in df.columns:
    df["trestbps"] = pd.to_numeric(df["trestbps"], errors='coerce')
if "chol" in df.columns:
    df["chol"] = pd.to_numeric(df["chol"], errors='coerce')

# recover only 1st (50) rec
n_convert = min(50, len(df))

# trestbps: from (kPa) to (mmHg)
if "trestbps" in df.columns:
    df.loc[:n_convert-1, "trestbps"] = df.loc[:n_convert-1, "trestbps"] / 0.133322

# chol: from (mmol/L) to (mg/dL)
if "chol" in df.columns:
    df.loc[:n_convert-1, "chol"] = df.loc[:n_convert-1, "chol"] / 0.02586

# transfotm sex col from male/female to 1/0, respectfully
if 'sex' in df.columns:
    # Use 'Male' and 'Female' due to prior standardization
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype('Int64', errors='ignore')

# transfotm fbs col from true/false to 1/0, respectfully
if 'fbs' in df.columns:
    # Convert 'True'/'False' strings (if present) to 1/0
    df['fbs'] = df['fbs'].apply(standardize_boolean_like)
    df['fbs'] = df['fbs'].map({'True': 1, 'False': 0}).fillna(df['fbs']).astype('Int64', errors='ignore')

# Final report and save for preparation phase (if starting from uncleaned data)
if not os.path.exists(INPUT_CSV):
     print(f"Skipping final report and save to {OUTPUT_CSV} as data was loaded from excel.")
else:
    report_basic_info(df, "cleaned and transformed")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved cleaned and transformed data to:", OUTPUT_CSV)

print("Transformation done.")


# Ensure the dataframe is loaded if the script started from the cleaned excel
if 'df' not in locals() or df.empty:
    file_path = "cleaned_data_v2_after chang(1).xlsx"
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        print(f"Loaded dataset from {file_path} for splitting.")
    else:
        print(f"Dataframe is empty or not loaded. Cannot proceed with splitting/modeling. Please check data source.")
        exit()

# ------------------------------------------------------- Data Splitting --------------------------------------------------

if "target" in df.columns:
    # Define features and target variable
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Display shape of the splits
    print("\n--- Data Splitting ---")
    print("Training set:", X_train.shape)
    print("Testing set:", X_test.shape)

    # Combine and save splits
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv('train_PR.csv', index=False)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv('test_PR.csv', index=False)
    print("Saved training data to: train_PR.csv")
    print("Saved testing data to: test_PR.csv")
else:
    print("Cannot perform data splitting: 'target' column not found.")
    exit() # Exit if no target column for modeling


# ------------------------------------------------------- PHASE 4 – EXPLORATORY DATA ANALYSIS (EDA) --------------------------------------------------

print("\n\n============================================")
print("3. PHASE 4 – EXPLORATORY DATA ANALYSIS (EDA)")
print("============================================\n")

# Create a copy for EDA and rename columns
data_EDA = df.copy()

data_EDA = data_EDA.rename(columns={
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest_Pain_Type',
    'trestbps': 'Resting_Blood_Pressure',
    'chol': 'Serum_Cholesterol',
    'fbs': 'Fasting_Blood_Sugar',
    'restecg': 'Resting_ECG_Results',
    'thalachh': 'Max_Heart_Rate',
    'exang': 'Exercise_Induced_Angina',
    'oldpeak': 'ST_Depression',
    'slope': 'ST_Slope',
    'ca': 'Major_Vessels',
    'thal': 'Thalassemia',
    'target': 'Heart_Disease'
})

important_cols = [
    'Age',
    'Chest_Pain_Type',
    'Resting_Blood_Pressure',
    'Serum_Cholesterol',
    'Max_Heart_Rate',
    'ST_Depression',
    'Major_Vessels',
    'Thalassemia'
]

# Ensure only columns present in the DataFrame are used
important_cols = [col for col in important_cols if col in data_EDA.columns]

print("\nEDA will focus on the following columns:")
print(important_cols)

# Outlier detection using IQR method

print("\n========== Outlier Detection (IQR) ==========")
outlier_counts = {}
for col in important_cols:
    Q1 = data_EDA[col].quantile(0.25)
    Q3 = data_EDA[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = data_EDA[(data_EDA[col] < lower) | (data_EDA[col] > upper)]
    count = ((data_EDA[col] < lower) | (data_EDA[col] > upper)).sum()
    outlier_counts[col] = count

    if not outliers.empty:
        print(f"\nOutliers in '{col}':")
        # Ensure the column exists before trying to print it
        if col in outliers.columns:
             print(outliers[[col]])
    else:
        print(f"\nNo outliers found in '{col}'.")


# Outlier Counts Table
outlier_df = pd.DataFrame(
    list(outlier_counts.items()),
    columns=['Column', 'Outlier_Count']
)
print("\nOutlier counts per column:")
print(outlier_df)

# create folder for visualizations
output_folder = "visualizations"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Boxplot for important columns
# Check if there are any important columns to plot
if important_cols:
    plt.figure(figsize=(17, 8))
    sns.boxplot(data=data_EDA[important_cols])
    plt.title('Boxplot for Important Indicators', fontsize=18)
    plt.xlabel('Indicators', fontsize=14)
    plt.ylabel('Values', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Histograms for each important column
    data_EDA[important_cols].hist(bins=15, figsize=(16, 10))
    plt.suptitle('Histograms for Each Indicator', fontsize=16)
    plt.show()

    # --- Individual Plots ---

    # Histogram – Serum Cholesterol
    if 'Serum_Cholesterol' in data_EDA.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data_EDA['Serum_Cholesterol'],
            bins=20,
            kde=True,
            color='skyblue',
            edgecolor='black'
        )
        plt.title('Distribution of Cholesterol Levels', fontsize=14)
        plt.xlabel('Cholesterol (mg/dl)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "histogram_cholesterol.png"), dpi=300)
        plt.close()

    # Boxplot – Resting Blood Pressure
    if 'Resting_Blood_Pressure' in data_EDA.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=data_EDA['Resting_Blood_Pressure'], color='lightgreen')
        plt.title('Box Plot of Resting Blood Pressure', fontsize=14)
        plt.xlabel('Resting Blood Pressure (mm Hg)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "boxplot_trestbps.png"), dpi=300)
        plt.close()

    # Scatter – Age vs Max Heart Rate
    if 'Age' in data_EDA.columns and 'Max_Heart_Rate' in data_EDA.columns:
        plt.figure(figsize=(7, 5))
        plt.scatter(
            data_EDA['Age'],
            data_EDA['Max_Heart_Rate'],
            color='coral',
            alpha=0.7
        )
        plt.title('Relationship between Age and Maximum Heart Rate', fontsize=14)
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Maximum Heart Rate', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "scatter_age_thalachh.png"), dpi=300)
        plt.close()

    # Scatter – Cholesterol vs Resting Blood Pressure
    if 'Serum_Cholesterol' in data_EDA.columns and 'Resting_Blood_Pressure' in data_EDA.columns:
        plt.figure(figsize=(7, 5))
        plt.scatter(
            data_EDA['Serum_Cholesterol'],
            data_EDA['Resting_Blood_Pressure'],
            color='mediumseagreen',
            alpha=0.7
        )
        plt.title('Relationship between Cholesterol and Resting Blood Pressure', fontsize=14)
        plt.xlabel('Cholesterol (mg/dl)', fontsize=12)
        plt.ylabel('Resting Blood Pressure (mm Hg)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "scatter_chol_trestbps.png"), dpi=300)
        plt.close()

    # Scatter – Age vs Cholesterol
    if 'Age' in data_EDA.columns and 'Serum_Cholesterol' in data_EDA.columns:
        plt.figure(figsize=(7, 5))
        plt.scatter(
            data_EDA['Age'],
            data_EDA['Serum_Cholesterol'],
            color='slateblue',
            alpha=0.7
        )
        plt.title('Relationship between Age and Cholesterol', fontsize=14)
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Cholesterol (mg/dl)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "scatter_age_chol.png"), dpi=300)
        plt.close()

    print("\n[EDA] All visualizations generated – check the 'visualizations' folder.")
else:
    print("\n[EDA] Not enough important columns found for visualization.")


# ------------------------------------------------------- PHASE 5 - Data Modeling --------------------------------------------------

print("\n\n============================================")
print("4. PHASE 5 – DATA MODELING")
print("============================================\n")

# X_train, X_test, y_train, y_test were prepared in the Data Splitting step.

# ============================ Models before tune ============================

# Initialize and train models

# 1) Random Forest
RandomForestClassifier_Model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
RandomForestClassifier_Model.fit(X_train, y_train)

# 2) Decision Tree
DecisionTreeClassifier_Model = DecisionTreeClassifier(random_state=42)
DecisionTreeClassifier_Model.fit(X_train, y_train)

# 3) Logistic Regression
LogisticRegression_Model = LogisticRegression(max_iter=1200)
LogisticRegression_Model.fit(X_train, y_train)

# 4) K-Nearest Neighbors
KNeighborsClassifier_Model = KNeighborsClassifier(n_neighbors=6)
KNeighborsClassifier_Model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    pred_y = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, pred_y),
        "Precision": precision_score(y_test, pred_y, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, pred_y, average='weighted', zero_division=0),
        "F1": f1_score(y_test, pred_y, average='weighted', zero_division=0)
    }


def plot_confusion_matrix(model, X_test, y_test, title):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Purples)
    plt.title(title)
    plt.show()



before_results = {
    "RandomForest": evaluate_model(RandomForestClassifier_Model, X_test, y_test),
    "DecisionTree": evaluate_model(DecisionTreeClassifier_Model, X_test, y_test),
    "LogisticRegression": evaluate_model(LogisticRegression_Model, X_test, y_test),
    "KNN": evaluate_model(KNeighborsClassifier_Model, X_test, y_test),
}

plot_confusion_matrix(RandomForestClassifier_Model, X_test, y_test, "RandomForest (Before)")
plot_confusion_matrix(DecisionTreeClassifier_Model, X_test, y_test, "DecisionTree (Before)")
plot_confusion_matrix(LogisticRegression_Model, X_test, y_test, "LogisticRegression (Before)")
plot_confusion_matrix(KNeighborsClassifier_Model, X_test, y_test, "KNN (Before)")


# ============================ Tuning ============================
models_for_tuning = {
    "RandomForest": RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    ),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1200),
    "KNN": KNeighborsClassifier()
}

param_grids = {
    "RandomForest": {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    },
    "DecisionTree": {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    },
    "LogisticRegression": {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    },
    "KNN": {
        'n_neighbors': [5, 7, 9],
        'weights': ['uniform', 'distance']
    }
}

tuned_models = {}

for name, model in models_for_tuning.items():
    grid = GridSearchCV(
        model,
        param_grids[name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    tuned_models[name] = grid.best_estimator_

after_results = {}
for name, model in tuned_models.items():
    after_results[name] = evaluate_model(model, X_test, y_test)


before_results = {
    "RandomForest": evaluate_model(RandomForestClassifier_Model, X_test, y_test),
    "DecisionTree": evaluate_model(DecisionTreeClassifier_Model, X_test, y_test),
    "LogisticRegression": evaluate_model(LogisticRegression_Model, X_test, y_test),
    "KNN": evaluate_model(KNeighborsClassifier_Model, X_test, y_test),
}

before_df = pd.DataFrame(before_results).T.reset_index().rename(columns={"index": "Model"})
before_df[["Accuracy","Precision","Recall","F1"]] = before_df[["Accuracy","Precision","Recall","F1"]].round(4)

print("\n===== Models before tuning =====\n")
print(before_df.to_string(index=False))



after_results = {}
for name, model in tuned_models.items():
    after_results[name] = evaluate_model(model, X_test, y_test)

after_df = pd.DataFrame(after_results).T.reset_index().rename(columns={"index": "Model"})
after_df[["Accuracy","Precision","Recall","F1"]] = after_df[["Accuracy","Precision","Recall","F1"]].round(4)

print("\n===== Models after tuning =====\n")
print(after_df.to_string(index=False))


plot_confusion_matrix(tuned_models["RandomForest"], X_test, y_test, "RandomForest (After)")
plot_confusion_matrix(tuned_models["DecisionTree"], X_test, y_test, "DecisionTree (After)")
plot_confusion_matrix(tuned_models["LogisticRegression"], X_test, y_test, "LogisticRegression (After)")
plot_confusion_matrix(tuned_models["KNN"], X_test, y_test, "KNN (After)")

print("\n(Data Preparation -> EDA -> Data Modeling) executed successfully.")
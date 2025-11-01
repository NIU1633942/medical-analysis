import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(BASE_DIR, "..", "results", "results_classification.csv")
#FEATURES_CSV = os.path.join(BASE_DIR, "..", "data", "input", "features.csv")
FEATURES_XLSX = os.path.join(BASE_DIR, "..", "data", "input", "features.xlsx")

MODELS_DIR = os.path.join(BASE_DIR, "..", "data", "output", "models")

#load values 
# X (n_samples, n_features), Y (n_samples, )
#df = pd.read_csv(FEATURES_CSV) #csv
df = pd.read_excel(FEATURES_XLSX, sheet_name='Sheet1') #excel
feature_cols = [c for c in df.columns if c.startswith('original_glcm_')]

X = df[feature_cols].values
Y = np.array([d for d in df['diagnosis']])
groups = df['nodule_id'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

#Define models
models = {
    "SVM": SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogReg": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    "RF": RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
}

# Train and evaluate model performance 
results = []
for name, clf in models.items():
    clf.fit(X_train, y_train)

    # Calibrar probabilidades (opcional)
    calibrated = CalibratedClassifierCV(clf, cv='prefit')
    calibrated.fit(X_train, y_train)

    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)[:, 1]

    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_prob)

    results.append([name, acc, prec, rec, f1, auc])

    print(f"\n=== {name} ===")
    print(metrics.classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
    print(f"AUC: {auc:.3f}")

    model_path = os.path.join(MODELS_DIR, f"{name}_model.joblib")
    joblib.dump(calibrated, model_path)

df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])
print("\n=== Summary of Models ===")
print(df.sort_values(by="AUC", ascending=False))
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
df.to_csv(RESULTS_CSV, index=False)
print("Pipeline finished. Models trained and evaluated.")
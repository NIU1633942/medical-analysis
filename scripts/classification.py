from typing import Counter
import numpy as np
import pandas as pd
import os
import itertools

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold 
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
FEATURES_CSV = os.path.join(BASE_DIR, "..", "results", "features.csv")
#FEATURES_XLSX = os.path.join(BASE_DIR, "..", "data", "input", "features.xlsx")

MODELS_DIR = os.path.join(BASE_DIR, "..", "data", "output", "models")

#load values 
# X (n_samples, n_features), Y (n_samples, )
df = pd.read_csv(FEATURES_CSV) #csv
#df = pd.read_excel(FEATURES_XLSX, sheet_name='Sheet1') #excel
feature_cols = [c for c in df.columns if c.startswith('original_glcm_')]

X = df[feature_cols].values
Y = np.array([d for d in df['diagnosis']])
groups = df['nodule_id'].values

#Define models
models = {
    "SVM": SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogReg": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    "RF": RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
}

models = {
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "LogReg": LogisticRegression,
    "MLP": MLPClassifier,
    "RF": RandomForestClassifier
}

params_SVM = {
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'C' : [0.1, 1, 10, 100],
    'gamma' : ['scale', 'auto', 0.01, 0.1, 1],
}
params_KNN = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
params_LogReg = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2', None],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [500, 1000, 2000],
}
params_MLP = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [500, 1000, 2000],
}
params_RF = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}

param_grids = {
    "SVM": params_SVM,
    "KNN": params_KNN,
    "LogReg": params_LogReg,
    "MLP": params_MLP,
    "RF": params_RF
}

def generate_model_instances(models, param_grids):
    model_instances = []

    for model_name, ModelClass in models.items():
        params = param_grids.get(model_name, {})
        if params:
            keys, values = zip(*params.items())
            for combination in itertools.product(*values):
                param_dict = dict(zip(keys, combination))
                # Crear instancia con esos parámetros
                instance = ModelClass(**param_dict)
                model_instances.append({
                    "model_name": model_name,
                    "model_instance": instance,
                    "params": param_dict
                })
        else:
            # Si no hay parámetros, solo instancia base
            instance = ModelClass()
            model_instances.append({
                "model_name": model_name,
                "model_instance": instance,
                "params": {}
            })

    return model_instances

def generate_model_filename(params):
    param_str = "_".join(f"{k}-{v}" for k, v in params.items())
    param_str = param_str.replace(" ", "").replace(":", "-").replace("/", "-")
    return f"{name}_{param_str}.joblib"

def evaluate_model(X, Y, clf, method="train_test_split", k_values=[5], test_size=0.3, random_state=42):
    results = []
    if method == "train_test_split":
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, stratify=Y
        )
        clf.fit(X_train, y_train)

        calibrated = CalibratedClassifierCV(clf, cv='prefit')
        calibrated.fit(X_train, y_train)

        y_pred = calibrated.predict(X_test)
        y_prob = calibrated.predict_proba(X_test)[:, 1]

        results.append({
            "Method": method,
            "K": None,
            "Accuracy": metrics.accuracy_score(y_test, y_pred),
            "Precision": metrics.precision_score(y_test, y_pred),
            "Recall": metrics.recall_score(y_test, y_pred),
            "F1": metrics.f1_score(y_test, y_pred),
            "AUC": metrics.roc_auc_score(y_test, y_prob)
        })
    elif method in ["kfold", "stratified_kfold"]:
        for k in k_values:
            if method == "stratified_kfold":
                kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
            else:
                kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

        acc, prec, rec, f1s, aucs = [], [], [], [], []
        for train_idx, test_idx in kf.split(X, Y if method=="stratified_kfold" else None):
            clf.fit(X[train_idx], Y[train_idx])
            calibrated = CalibratedClassifierCV(clf, cv='prefit')
            calibrated.fit(X[train_idx], Y[train_idx])

            y_pred = calibrated.predict(X[test_idx])
            y_prob = calibrated.predict_proba(X[test_idx])[:, 1]

            acc.append(metrics.accuracy_score(Y[test_idx], y_pred))
            prec.append(metrics.precision_score(Y[test_idx], y_pred))
            rec.append(metrics.recall_score(Y[test_idx], y_pred))
            f1s.append(metrics.f1_score(Y[test_idx], y_pred))
            aucs.append(metrics.roc_auc_score(Y[test_idx], y_prob))

        results.append({
            "Method": method,
            "K": k,
            "Accuracy": np.mean(acc),
            "Accuracy_std": np.std(acc),
            "Precision": np.mean(prec),
            "Recall": np.mean(rec),
            "F1": np.mean(f1s),
            "AUC": np.mean(aucs)
            })
    return results





all_model_instances = generate_model_instances(models, param_grids)
print(f"Total model combinations to evaluate: {len(all_model_instances)}")
class_counts = Counter(Y)
min_samples = min(class_counts.values())
k_values = [3, 5, 7, 10]
k_values_safe = [k for k in k_values if k <= min_samples]

results = []

for model_info in all_model_instances:
    name = model_info["model_name"]
    clf = model_info["model_instance"]
    params = model_info["params"]
    print(f"\nTraining model: {name} with params: {params}")

    # Train/Test Split
    print(f"\nMethod = train_test_split")
    r_tt = evaluate_model(X, Y, clf, method="train_test_split")
    for r in r_tt:
        r.update({"Model": name, "Params": params})
        results.append(r)

    # K-Fold
    print(f"\nMethod = kfold")
    r_kf = evaluate_model(X, Y, clf, method="kfold", k_values=k_values)
    for r in r_kf:
        r.update({"Model": name, "Params": params})
        results.append(r)

    # Stratified K-Fold
    print(f"\nMethod = stratified_kfold")
    r_skf = evaluate_model(X, Y, clf, method="stratified_kfold", k_values=k_values_safe)
    for r in r_skf:
        r.update({"Model": name, "Params": params})
        results.append(r)

# Guardar resultados
df_results = pd.DataFrame(results)
print("\n=== Summary of Models ===")
print(df_results.sort_values(by="AUC", ascending=False))
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
df_results.to_csv(RESULTS_CSV, index=False)
print("Pipeline finished. Models trained and evaluated.")

"""
# Train and evaluate model performance 
for model_info in all_model_instances:
    name = model_info["model_name"]
    clf = model_info["model_instance"]
    params = model_info["params"]
    print(f"\nTraining model: {name} with params: {params}")

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

    results.append({
        "Model": name,
        "Params": params,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc
    })

    print(f"\n=== {name} ===")
    print(metrics.classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))

    #model_filename = generate_model_filename(params)
    #model_path = os.path.join(MODELS_DIR, model_filename)
    #joblib.dump(calibrated, model_path)

df_results = pd.DataFrame(results)
print("\n=== Summary of Models ===")
print(df_results.sort_values(by="AUC", ascending=False))
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
df_results.to_csv(RESULTS_CSV, index=False)
print("Pipeline finished. Models trained and evaluated.")
"""
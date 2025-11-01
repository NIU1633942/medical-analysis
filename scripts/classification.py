import numpy as np
import pandas as pd
import os
import itertools

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
    'n_neighbors': [3, 5, 7, 9],
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

all_model_instances = generate_model_instances(models, param_grids)
print(f"Total model combinations to evaluate: {len(all_model_instances)}")

results = []
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
    print(f"AUC: {auc:.3f}")

    model_filename = generate_model_filename(params)
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(calibrated, model_path)

df_results = pd.DataFrame(results)
print("\n=== Summary of Models ===")
print(df_results.sort_values(by="AUC", ascending=False))
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
df_results.to_csv(RESULTS_CSV, index=False)
print("Pipeline finished. Models trained and evaluated.")
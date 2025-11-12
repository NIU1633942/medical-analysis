import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(BASE_DIR, "..", "results", "results_classification.csv")
PLOT_DIR = os.path.join(BASE_DIR, "..", "results", "plots")
df = pd.read_csv(RESULTS_CSV)

# Assign column names (based on your description)
df.columns = [
    "Method", "K", "Accuracy", "Precision", "Recall", "F1", "AUC",
    "CM", "Model", "Params", "Accuracy_std"
]

df["K"] = pd.to_numeric(df["K"], errors="coerce")
df["Accuracy"] = df["Accuracy"].astype(float)
df["Accuracy_std"] = df["Accuracy_std"].astype(float)
df["Precision"] = df["Precision"].astype(float)
df["Recall"] = df["Recall"].astype(float)
df["F1"] = df["F1"].astype(float)
df["AUC"] = df["AUC"].astype(float)

# Convert stringified lists/dicts
df["CM"] = df["CM"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df["Params"] = df["Params"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

df["Params_str"] = df["Params"].apply(lambda x: str(x))


def plot_metrics_comparison():
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    df_melted = df.melt(id_vars=["Method", "Model"], value_vars=metrics_cols, var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    for metric in metrics_cols:
        subset = df[df["Method"] == "train_test_split"]
    plt.title("Model Performance by Evaluation Method")
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Method")
    plt.ylim(0, 1)
    plt.legend(title="Method")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.show()

def accuracy_k_cross_validation():
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df[df["Method"].str.contains("kfold", case=False, na=False)], x="K", y="Accuracy", hue="Method", marker="o")
    plt.title("Accuracy vs Number of Folds (K)")
    plt.xlabel("K (number of folds)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def confunsion_matrix():
    df["TP"] = df["CM"].apply(lambda x: x[3])
    df["FP"] = df["CM"].apply(lambda x: x[1])
    df["FN"] = df["CM"].apply(lambda x: x[2])
    df["TN"] = df["CM"].apply(lambda x: x[0])

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Method", y="TP", hue="Model")
    plt.title("True Positives by Method")
    plt.show()

def metric_comparison_by_model():
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    df_melted = df.melt(id_vars=["Method", "Model"], value_vars=metrics_cols, var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x="Metric", y="Score", hue="Model")
    plt.title("Model Comparison Across Metrics")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()

def plot_model_metrics():
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    for model in df["Model"].unique():
        subset = df[df["Model"] == model]

        print(f"Generando gráficos para modelo: {model}")

        for metric in metrics_cols:
            plt.figure(figsize=(8, 5))
            sns.barplot(
                data=subset,
                x="Params_str",
                y=metric,
                palette="Set2"
            )
            plt.title(f"{model} - {metric} por Parámetros", fontsize=14)
            plt.ylabel(metric)
            plt.xlabel("Parámetros")
            plt.xticks(rotation=30, ha="right")
            plt.ylim(0, 1)
            plt.tight_layout()

            # Guardar gráfico como PNG
            filename = f"{model}_{metric}.png".replace(" ", "_")
            file_path = os.path.join(PLOT_DIR, filename)
            plt.savefig(file_path, dpi=150)
            #plt.show()
    
def plot_confusion_matrices():
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Loop through each model
    for model in df["Model"].unique():
        subset = df[df["Model"] == model]

        for _, row in subset.iterrows():
            cm_flat = row["CM"]
            cm = np.array(cm_flat).reshape(2,2)  # TN, FP, FN, TP
            params_str = row["Params_str"]

            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
            plt.title(f"{model} - Confusion Matrix\nParams: {params_str}", fontsize=12)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.xticks([0.5, 1.5], ["Negative", "Positive"])
            plt.yticks([0.5, 1.5], ["Negative", "Positive"], rotation=0)
            plt.tight_layout()

            # Save figure
            filename = f"{model}_CM_{params_str}.png".replace(" ", "_").replace("{","").replace("}","").replace("'","").replace(",","_")
            file_path = os.path.join(PLOT_DIR, filename)
            plt.savefig(file_path, dpi=150)
            #plt.show()

#plot_metrics_comparison()
#accuracy_k_cross_validation()
#confunsion_matrix()
#metric_comparison_by_model()

plot_confusion_matrices()
plot_model_metrics()

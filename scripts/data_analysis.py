import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import numpy as np
from sklearn import metrics
from collections import Counter

from sklearn.metrics import (
    roc_curve, auc
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CLASSIFICATION_CSV = os.path.join(BASE_DIR, "..", "results", "results_classification.csv")
RESULTS_SEGMENTATION_CSV = os.path.join(BASE_DIR, "..", "results", "results_segmentation.csv")
PLOT_DIR = os.path.join(BASE_DIR, "..", "results", "plots")
df_classification = pd.read_csv(RESULTS_CLASSIFICATION_CSV)
df_segmentation = pd.read_csv(RESULTS_SEGMENTATION_CSV)

# Assign column names (based on your description)
df_classification.columns = [
    "Method", "K", "Accuracy", "Precision", "Recall", "F1", "AUC",
    "CM", "Model", "Params", "Accuracy_std"
]

df_classification["K"] = pd.to_numeric(df_classification["K"], errors="coerce")
df_classification["Accuracy"] = df_classification["Accuracy"].astype(float)
df_classification["Accuracy_std"] = df_classification["Accuracy_std"].astype(float)
df_classification["Precision"] = df_classification["Precision"].astype(float)
df_classification["Recall"] = df_classification["Recall"].astype(float)
df_classification["F1"] = df_classification["F1"].astype(float)
df_classification["AUC"] = df_classification["AUC"].astype(float)

# Convert stringified lists/dicts
df_classification["CM"] = df_classification["CM"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_classification["Params"] = df_classification["Params"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

df_segmentation["threshold_method"] = df_segmentation["threshold_method"].astype(str)


def select_best_dice_per_image():
    """
    Given a DataFrame with columns:
        image, sigma, median_size, threshold_method, threshold_value,
        open_size, close_size, dice

    Returns a DataFrame containing ONLY the best Dice configuration
    for each image.
    """

    # group by image and select the row with max dice
    best_df = (
        df_segmentation.loc[df_segmentation.groupby("image")["dice"].idxmax()]
        .reset_index(drop=True)
        .sort_values("image")
    )

    return best_df

def plot_global_dice_distribution():
    dice = df_segmentation["dice"]

    mean, std = dice.mean(), dice.std()
    min_val, max_val = dice.min(), dice.max()

    plt.figure(figsize=(8, 5))
    plt.hist(dice, bins=30)
    plt.title("Global Dice Distribution")
    plt.xlabel("Dice Coefficient")
    plt.ylabel("Frequency")

    # Text box with statistics
    text = f"Mean = {mean:.3f}\nStd = {std:.3f}\nMin = {min_val:.3f}\nMax = {max_val:.3f}"
    plt.gca().text(0.98, 0.95, text, transform=plt.gca().transAxes,
                   verticalalignment="top", horizontalalignment="right",
                   bbox=dict(facecolor="white", alpha=0.7))

    filename = f"global_dice_distribution.png".replace(" ", "_")
    file_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {file_path}")

def print_acceptable_segmentations():
    total = len(df_segmentation)
    count_07 = (df_segmentation["dice"] > 0.7).sum()
    percent_07 = (count_07 / total) * 100

    print("\n📌 ACCEPTABLE SEGMENTATIONS (Dice > 0.7)")
    print("-------------------------------------------------")
    print(f"Count: {count_07} / {total}")
    print(f"Percentage: {percent_07:.2f}%")
    print("-------------------------------------------------\n")

def print_outlier_segmentations():
    total = len(df_segmentation)
    outliers = df_segmentation[df_segmentation["dice"] < 0.2]
    count_out = len(outliers)
    percent_out = (count_out / total) * 100

    print("\n⚠️ OUTLIER SEGMENTATIONS (Dice < 0.2)")
    print("-------------------------------------------------")
    print(f"Count: {count_out} / {total}")
    print(f"Percentage: {percent_out:.2f}%")
    print("\nList of outlier images:")
    print(outliers[["image", "dice"]])
    print("-------------------------------------------------\n")

def most_frequent_best_segmentations(top_n=10):
    """
    Analyze which parameter combinations appear most often in the best segmentations per image.

    Args:
        df_segmentation (pd.DataFrame): DataFrame with columns ['image', 'sigma', 'median_size', 
                                      'threshold_method', 'threshold_value', 'open_size', 'close_size', 'dice']
        top_n (int): Number of top parameter sets to show.

    Returns:
        pd.DataFrame: Top parameter combinations with counts.
    """

    # 1. Select the best segmentation per image
    best_per_image = df_segmentation.loc[df_segmentation.groupby("image")["dice"].idxmax()]

    # 2. Create a string representation of parameter sets
    param_cols = ["sigma", "median_size", "threshold_method", "threshold_value", "open_size", "close_size"]
    best_per_image["param_set"] = best_per_image[param_cols].astype(str).agg("-".join, axis=1)

    # 3. Count frequency of each parameter set
    counter = Counter(best_per_image["param_set"])
    most_common = counter.most_common(top_n)

    # 4. Convert to DataFrame
    top_df = pd.DataFrame(most_common, columns=["param_set", "count"])

    return top_df


def mean_dice_per_parameter_combination():
    group_cols = ["sigma","median_size","threshold_method",
                  "threshold_value","open_size","close_size"]

    mean_combinations = (
        df_segmentation.groupby(group_cols)["dice"]
        .mean().reset_index()
    )

    plt.figure(figsize=(10,6))
    sns.histplot(mean_combinations["dice"], bins=30, kde=True)
    plt.title("Distribution of Mean Dice Across Parameter Combinations")
    plt.xlabel("Mean Dice")

    filename = f"mean_dice_distribution.png".replace(" ", "_")
    file_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {file_path}")

def shorten_params(params_dict):
    """Convert parameter dict to compact string like 'C=5|kernel=linear'."""
    if not isinstance(params_dict, dict):
        return str(params_dict)
    return "|".join([f"{k}={v}" for k, v in params_dict.items()])

df_classification["Params_str"] = df_classification["Params"].apply(shorten_params)


def assign_param_codes(df):
    unique_params = df["Params_str"].unique()
    param_mapping = {p: f"P{i+1}" for i, p in enumerate(unique_params)}
    df["ParamCode"] = df["Params_str"].map(param_mapping)
    return param_mapping

# Plot all metrics per model
def plot_model_metrics():
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    for model in df_classification["Model"].unique():
        subset = df_classification[df_classification["Model"] == model].copy()
        param_mapping = assign_param_codes(subset)
        print(f"Generando gráficos para modelo: {model}")

        for metric in metrics_cols:
            plt.figure(figsize=(8, 5))
            sns.barplot(
                data=subset,
                x="ParamCode",
                y=metric,
                palette="Set2"
            )
            plt.title(f"{model} - {metric} por Parámetros", fontsize=14)
            plt.ylabel(metric)
            plt.xlabel("Parameter Set (see below)")
            
            y_min = max(0, subset[metric].min() - 0.02)
            y_max = min(1, subset[metric].max() + 0.02)
            plt.ylim(y_min, y_max)

            # Show param legend under the plot
            legend_text = "\n".join([f"{code}: {full}" for full, code in param_mapping.items()])
            plt.figtext(0.01, -0.05, legend_text, ha="left", fontsize=8, wrap=True)

            plt.tight_layout()

            filename = f"{model}_{metric}.png".replace(" ", "_")
            file_path = os.path.join(PLOT_DIR, filename)
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved: {file_path}")

# Plot confusion matrices per model
def plot_confusion_matrices():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    for model in df_classification["Model"].unique():
        subset = df_classification[df_classification["Model"] == model].copy()
        param_mapping = assign_param_codes(subset)

        for _, row in subset.iterrows():
            cm_flat = row["CM"]
            cm = np.array(cm_flat).reshape(2,2)  # TN, FP, FN, TP
            params_code = row["ParamCode"]

            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
            plt.title(f"{model} - Confusion Matrix\nParams: {params_code}", fontsize=12)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.xticks([0.5, 1.5], ["Negative", "Positive"])
            plt.yticks([0.5, 1.5], ["Negative", "Positive"], rotation=0)
            plt.tight_layout()

            filename = f"{model}_CM_{params_code}.png".replace(" ", "_")
            file_path = os.path.join(PLOT_DIR, filename)
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved: {file_path}")

# Plot top 3 parameters per model & method
def plot_top3_params_per_model():
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    methods = ["train_test_split", "kfold", "stratified_kfold"]

    sns.set_theme(style="whitegrid", font_scale=1.1)

    for model in df_classification["Model"].unique():
        df_model = df_classification[df_classification["Model"] == model].copy()
        param_mapping = assign_param_codes(df_model)
        print(f"\nGenerating plots for model: {model}")

        for metric in metrics:
            plt.figure(figsize=(15, 5))

            for i, method in enumerate(methods, start=1):
                df_subset = df_model[df_model["Method"] == method]

                if df_subset.empty:
                    continue

                top3 = df_subset.nlargest(3, metric)

                plt.subplot(1, 3, i)
                sns.barplot(
                    data=top3,
                    x="ParamCode",
                    y=metric,
                    palette="Set2"
                )
                plt.title(f"{method}\nTop 3 Params by {metric}")
                plt.xlabel("Parameter Set (see below)")
                plt.ylabel(metric)

                y_min = max(0, df_model[metric].min() - 0.02)
                y_max = min(1, df_model[metric].max() + 0.02)
                plt.ylim(y_min, y_max)

                # Show metric value on top of bars
                for index, row in enumerate(top3.itertuples()):
                    plt.text(
                        index, getattr(row, metric) + 0.02,
                        f"{getattr(row, metric):.3f}",
                        ha='center', fontsize=9
                    )

                # Show param legend under the subplot
                legend_text = "\n".join([f"{code}: {full}" for full, code in param_mapping.items()])
                plt.figtext(0.01, -0.05, legend_text, ha="left", fontsize=8, wrap=True)

            plt.suptitle(f"{model} — Top 3 Parameter Sets by {metric}", fontsize=14, y=1.05)
            plt.tight_layout()

            filename = f"{model}_Top3_{metric}.png".replace(" ", "_")
            file_path = os.path.join(PLOT_DIR, filename)
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved: {file_path}")

def plot_model_param_performance_across_methods():
    """
    For every metric, shows how each model+parameter set performed
    across different training methods (train_test_split, kfold, stratified_kfold),
    selecting the best K performance for each method.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    # Pick best (max) value per Model, Params_str, and Method (since K can vary)
    best_df = (
        df_classification.sort_values(by=["Model", "Params_str", "Method", "K"])
          .groupby(["Model", "Params_str", "Method"], as_index=False)
          .agg({
              "K": "max",
              "Accuracy": "max",
              "Precision": "max",
              "Recall": "max",
              "F1": "max",
              "AUC": "max",
              "Accuracy_std": "mean"
          })
    )

    sns.set_theme(style="whitegrid", font_scale=1.1)

    for model in best_df["Model"].unique():
        df_model = best_df[best_df["Model"] == model].copy()
        print(f"\n📊 Generating plots for model: {model}")

        # Use Params_str to generate short codes
        param_mapping = assign_param_codes(df_model)

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df_model,
                x="ParamCode",
                y=metric,
                hue="Method",
                palette="Set2"
            )
            plt.title(f"{model} — {metric} across Training Methods", fontsize=14)
            plt.ylabel(metric)
            plt.xlabel("Parameter Set (see below)")
            
            y_min = max(0, df_model[metric].min() - 0.02)
            y_max = min(1, df_model[metric].max() + 0.02)
            plt.ylim(y_min, y_max)
            
            plt.legend(title="Training Method", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Show param mapping under plot
            legend_text = "\n".join([f"{code}: {full}" for full, code in param_mapping.items()])
            plt.figtext(0.01, -0.05, legend_text, ha="left", fontsize=8, wrap=True)

            plt.tight_layout()
            filename = f"{model}_{metric}_method_comparison.png".replace(" ", "_")
            file_path = os.path.join(PLOT_DIR, filename)
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved plot: {file_path}")

def plot_metric_distributions():
    metrics=["Accuracy", "Precision", "Recall", "F1", "AUC"]

    sns.set_theme(style="whitegrid", font_scale=1.1)
    for model in df_classification["Model"].unique():
        subset = df_classification[df_classification["Model"] == model].copy()
        print(f"\n📊 Plotting distributions for model: {model}")

        param_mapping = assign_param_codes(subset)

        for metric in metrics:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=subset, x="ParamCode", y=metric, palette="Set2")
            plt.title(f"{model} - {metric} Distribution Across Folds/Params", fontsize=14)
            plt.ylabel(metric)
            plt.xlabel("Parameter Set")

            y_min = max(0, subset[metric].min() - 0.02)
            y_max = min(1, subset[metric].max() + 0.02)
            plt.ylim(y_min, y_max)

            plt.tight_layout()

            filename = f"{model}_{metric}_boxplot.png".replace(" ", "_")
            plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved: {filename}")

def plot_roc_curves(top_n=3):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    for model in df_classification["Model"].unique():
        subset = df_classification[df_classification["Model"] == model].copy()

        # Select top N parameter sets by Accuracy
        top_params = subset.nlargest(top_n, "Accuracy")
        plt.figure(figsize=(8, 6))

        for row in top_params.itertuples():
            y_true = getattr(row, "all_y_true", None)
            y_prob = getattr(row, "all_y_prob", None)

            if y_true is None or y_prob is None:
                continue  # skip if predictions not stored

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{row.ParamCode} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"{model} — ROC Curves Top {top_n} Param Sets")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()

        filename = f"{model}_ROC_top{top_n}.png".replace(" ", "_")
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {filename}")

def plot_radar_top_params():
    metrics=["Accuracy", "Precision", "Recall", "F1", "AUC"]
    for model in df_classification["Model"].unique():
        subset = df_classification[df_classification["Model"] == model].copy()
        param_mapping = assign_param_codes(subset)
        # Choose top param by Accuracy
        top_row = subset.nlargest(1, "Accuracy").iloc[0]
        values = [top_row[m] for m in metrics]
        values += values[:1]  # close the loop

        # Angles for radar (close the loop separately)
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles_loop = np.concatenate((angles, [angles[0]]))  # for plotting

        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6,6))
        ax.plot(angles_loop, values, "o-", linewidth=2)
        ax.fill(angles_loop, values, alpha=0.25)

        # Use only original angles for labeling
        ax.set_thetagrids(angles * 180 / np.pi, metrics)

        # Dynamic radial limits
        min_val = max(0, min(values) - 0.02)
        max_val = min(1, max(values) + 0.02)
        ax.set_ylim(min_val, max_val)

        ax.set_title(f"{model} — Top Param Set Radar Chart ({top_row.ParamCode})", fontsize=12)

        filename = f"{model}_radar_top_param.png".replace(" ", "_")
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {filename}")

print(select_best_dice_per_image())
mean_dice_per_parameter_combination()
plot_global_dice_distribution()
#print_acceptable_segmentations()
#print_outlier_segmentations()
print(most_frequent_best_segmentations(top_n=5))

#plot_top3_params_per_model()
#plot_confusion_matrices()
#plot_model_metrics()
#plot_model_param_performance_across_methods()
#plot_metric_distributions()
#plot_roc_curves()
#plot_radar_top_params()

"""
Explication of every plot
1 plot_top3_params_per_model()

What it does:
For each model and training method, it selects the top 3 parameter sets based on a chosen metric (e.g., Accuracy).
It plots barplots of the selected metric values for these top 3 configurations.

Information you can obtain:
Which parameter sets perform best for each model.
How performance changes depending on the training method (train/test split, kfold, stratified kfold).
Helps quickly identify the most promising hyperparameter configurations.

Use case:
“Which combination of hyperparameters gives the best Accuracy/Precision/F1 for this model?”

2 plot_confusion_matrices()

What it does:
Plots the confusion matrix for each model and parameter set.
Confusion matrix shows True Positives, False Positives, True Negatives, False Negatives.

Information you can obtain:
Insight into where the model makes errors (e.g., more false positives vs. false negatives).
Can help identify if a model is biased towards one class.
Useful when Accuracy alone is misleading, especially for imbalanced datasets.

Use case:
“Even if Accuracy is high, is the model misclassifying many positives as negatives?”

3 plot_model_metrics()

What it does:
Plots all metrics (Accuracy, Precision, Recall, F1, AUC) for each parameter set of a model.
Each parameter set is encoded as a short code (P1, P2, …), with a legend mapping to full params.

Information you can obtain:
Compare different hyperparameter configurations across all metrics.
Quickly see trade-offs: e.g., a parameter set with high Accuracy might have lower Recall.
Useful for multi-metric optimization rather than focusing on a single metric.

Use case:
“Which parameter set balances Accuracy, Precision, and Recall best?”

4 plot_model_param_performance_across_methods()

What it does:
Compares the same model+parameter set across different training methods (train_test_split, kfold, stratified_kfold).
Plots bars for each metric to show how training methodology affects performance.

Information you can obtain:
Check stability of the model across training techniques.
See if performance varies a lot with kfold splits or is consistent.
Identify if a parameter set is robust or only works well in a specific setup.

Use case:
“Does this model + hyperparameter perform consistently across training strategies?”

5 plot_metric_distributions()

What it does:
Plots boxplots of metric values for each parameter set.
Shows distribution of metrics across folds (for kfold/stratified kfold) or multiple runs.

Information you can obtain:
Visualize variance and stability of a model’s performance.
See which parameters are more stable and less sensitive to train/test splits.
Identify outliers (folds where performance is particularly low or high).

Use case:
“Which hyperparameters give consistent Accuracy across folds?”

6 plot_roc_curves()

What it does:
Plots ROC curves for models and parameter sets.
Shows the trade-off between True Positive Rate (Recall) and False Positive Rate.
The AUC (Area Under Curve) is often shown as a performance summary.

Information you can obtain:
Compare model ability to separate positive vs negative classes.
Evaluate classification threshold performance.
ROC curves are especially useful for imbalanced datasets, where Accuracy can be misleading.

Use case:
“Does this model discriminate well between classes? Which hyperparameter set maximizes AUC?”

7 plot_radar_top_params()

What it does:
Plots radar/spider charts for the top parameter set of each model.
Shows multiple metrics (Accuracy, Precision, Recall, F1, AUC) in a single plot.

Information you can obtain:
Visual balance of a model’s performance across metrics.
Identify strengths and weaknesses in one glance.
Compare top parameter sets without cluttering multiple barplots.

Use case:
“Which top parameter set performs consistently across all metrics?”
"""
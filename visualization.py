# visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def plot_accuracy_vs_k(k_values, csv_files):
    best_scores = []
    for file in csv_files:
        df = pd.read_csv(file)
        best_score = df['mean_test_score'].max()
        best_scores.append(best_score)

    plt.figure(figsize=(16, 8))
    plt.plot(k_values, best_scores, marker='o', linestyle='-', color='blue')
    plt.title('Model Accuracy vs. Number of Selected Features (k)')
    plt.xlabel('Number of Features (k)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.xticks(k_values)
    plt.ylim(0.3, 0.6)
    plt.show()

def show_sample_predictions(y_test, y_pred, class_names, n=10):
    indices = np.random.choice(len(y_test), n, replace=False)
    for i in indices:
        true_idx = y_test[i]
        pred_idx = y_pred[i]
        true_label = class_names[true_idx] if true_idx < len(class_names) else f"Unknown ({true_idx})"
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Unknown ({pred_idx})"
        print(f"Sample {i}: True → {true_label}, Predicted → {pred_label}")

def plot_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(16, 8))
    disp.plot(xticks_rotation=45, cmap='Blues', ax=ax)
    plt.title("Confusion Matrix")
    plt.show()

def plot_accuracy_vs_c(df, kernel='rbf', tol='0.0001'):
    subset = df[df['params'].str.contains(f"kernel': '{kernel}'") & df['params'].str.contains(f"tol': {tol}")].copy()
    subset['C'] = subset['params'].str.extract(r"'C': ([\d\.]+)").astype(float)

    grouped = subset.groupby('C')['mean_test_score'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(grouped['C'], grouped['mean_test_score'], marker='o')
    plt.title(f"Avg Validation Accuracy vs. C (kernel='{kernel}', tol={tol})")
    plt.xlabel("C")
    plt.ylabel("Mean Validation Accuracy")
    plt.grid(True)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(subset['C'], subset['mean_test_score'], alpha=0.7)
    plt.title(f"Validation Accuracy vs. C (kernel='{kernel}', tol={tol})")
    plt.xlabel("C")
    plt.ylabel("Mean Validation Accuracy")
    plt.grid(True)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

def plot_accuracy_by_kernel(df):
    df['kernel'] = df['params'].str.extract(r"'kernel': '(\w+)'")
    kernel_perf = df.groupby('kernel')['mean_test_score'].mean().sort_values()

    kernel_perf.plot(kind='bar', color='skyblue', figsize=(6, 4))
    plt.title("Average Accuracy by Kernel Type")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Kernel")
    plt.tight_layout()
    plt.show()

def plot_hyperparam_heatmap(df):
    df['C'] = df['params'].str.extract(r"'C': ([\d\.]+)").astype(float)
    df['gamma'] = df['params'].str.extract(r"'gamma': '?(scale|auto|[\d\.]+)'?")[0]
    df['gamma'] = df['gamma'].replace({'scale': -1, 'auto': -2}).astype(float)

    heatmap_data = df.pivot_table(index='C', columns='gamma', values='mean_test_score')

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="Blues")
    plt.title("Validation Accuracy Heatmap (C vs Gamma)")
    plt.ylabel("C")
    plt.xlabel("Gamma")
    plt.tight_layout()
    plt.show()
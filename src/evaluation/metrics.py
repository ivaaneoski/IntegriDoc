import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_metrics(y_true, y_pred, y_probs=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
        except ValueError:
            metrics["roc_auc"] = None
            
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["REAL", "TAMPERED"], yticklabels=["REAL", "TAMPERED"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

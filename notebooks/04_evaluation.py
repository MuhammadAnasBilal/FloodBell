"""
Step 5: Model Evaluation -- Classification & Regression Metrics
===============================================================
- Regression: R2, MAE, RMSE
- Binary classification (threshold 0.5): Accuracy, Precision, Recall, Specificity, F1
- ROC curves + AUC, Confusion matrices, Threshold analysis
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, precision_recall_curve)
import joblib

warnings.filterwarnings('ignore')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  STEP 5: Model Evaluation")
print("=" * 60)

X_test_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'), index_col='id')
X_test_raw = pd.read_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'), index_col='id')
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'), index_col='id').squeeze()

models = {}
for name, fname, X_data in [('Ridge', 'ridge_model.pkl', X_test_scaled),
                              ('Random Forest', 'random_forest_model.pkl', X_test_raw),
                              ('XGBoost', 'xgboost_model.pkl', X_test_raw)]:
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        models[name] = (joblib.load(path), X_data)
        print(f"  Loaded {name}")

if not models:
    print("  ERROR: No models found. Run 02_ml_models.py first.")
    exit()

THRESHOLD = 0.5
y_binary = (y_test >= THRESHOLD).astype(int)

print("\n" + "-" * 60)
all_metrics = {}
all_preds = {}

for name, (model, X_data) in models.items():
    y_pred = model.predict(X_data)
    y_pred_binary = (y_pred >= THRESHOLD).astype(int)
    all_preds[name] = y_pred
    tn, fp, fn, tp = confusion_matrix(y_binary, y_pred_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr, tpr, _ = roc_curve(y_binary, y_pred)
    roc_auc = auc(fpr, tpr)
    m = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'accuracy': accuracy_score(y_binary, y_pred_binary),
        'precision': precision_score(y_binary, y_pred_binary, zero_division=0),
        'recall': recall_score(y_binary, y_pred_binary, zero_division=0),
        'specificity': specificity,
        'f1': f1_score(y_binary, y_pred_binary, zero_division=0),
        'auc': roc_auc,
        'fpr': fpr, 'tpr': tpr,
    }
    all_metrics[name] = m
    print(f"\n  {name}:")
    print(f"    Regression:     R2={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")
    print(f"    Classification: Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  Spec={m['specificity']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

metrics_save = {
    'models': list(all_metrics.keys()),
    'r2': [m['r2'] for m in all_metrics.values()],
    'mae': [m['mae'] for m in all_metrics.values()],
    'rmse': [m['rmse'] for m in all_metrics.values()],
    'accuracy': [m['accuracy'] for m in all_metrics.values()],
    'precision': [m['precision'] for m in all_metrics.values()],
    'recall': [m['recall'] for m in all_metrics.values()],
    'f1': [m['f1'] for m in all_metrics.values()],
    'auc': [m['auc'] for m in all_metrics.values()],
}
with open(os.path.join(OUTPUT_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(metrics_save, f, indent=2)

# Confusion Matrices
print("\n[2] Plotting confusion matrices...")
fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
if len(models) == 1: axes = [axes]
colors_cm = ['Blues', 'Greens', 'Oranges']
for i, (name, (model, X_data)) in enumerate(models.items()):
    y_pred_b = (model.predict(X_data) >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_binary, y_pred_b)
    sns.heatmap(cm, annot=True, fmt='d', cmap=colors_cm[i], ax=axes[i],
                xticklabels=['Safe', 'At-Risk'], yticklabels=['Safe', 'At-Risk'])
    axes[i].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.suptitle(f'Confusion Matrices (Threshold={THRESHOLD})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

# ROC Curves
print("[3] Plotting ROC curves...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = ['#2196F3', '#4CAF50', '#FF5722']
ax = axes[0]
for i, (name, m) in enumerate(all_metrics.items()):
    ax.plot(m['fpr'], m['tpr'], color=colors[i], lw=2, label=f"{name} (AUC={m['auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

ax = axes[1]
for i, (name, (model, X_data)) in enumerate(models.items()):
    y_pred = model.predict(X_data)
    prec, rec, _ = precision_recall_curve(y_binary, y_pred)
    ax.plot(rec, prec, color=colors[i], lw=2, label=f"{name}")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.suptitle('Classification Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_pr_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Threshold Analysis
print("[4] Threshold analysis...")
best_model_name = max(all_metrics, key=lambda x: all_metrics[x]['auc'])
best_model, best_X = models[best_model_name]
y_pred_best = best_model.predict(best_X)
thresholds = np.arange(0.3, 0.7, 0.02)
mt = {'threshold': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
for t in thresholds:
    yb = (y_pred_best >= t).astype(int)
    yt = (y_test >= t).astype(int)
    mt['threshold'].append(t)
    mt['precision'].append(precision_score(yt, yb, zero_division=0))
    mt['recall'].append(recall_score(yt, yb, zero_division=0))
    mt['f1'].append(f1_score(yt, yb, zero_division=0))
    mt['accuracy'].append(accuracy_score(yt, yb))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, mt['precision'], 'o-', label='Precision', color='#2196F3')
ax.plot(thresholds, mt['recall'], 's-', label='Recall', color='#4CAF50')
ax.plot(thresholds, mt['f1'], '^-', label='F1-Score', color='#FF5722')
ax.plot(thresholds, mt['accuracy'], 'D-', label='Accuracy', color='#9C27B0')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title(f'Threshold Analysis -- {best_model_name}', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("  Model Evaluation Complete!")
print("=" * 60)

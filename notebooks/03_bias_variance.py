"""
Step 4: Bias-Variance Tradeoff & Regularization Analysis
=========================================================
- Learning curves for all 3 models
- Validation curves for key hyperparameters
- Regularization effect analysis
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
import joblib

warnings.filterwarnings('ignore')
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  STEP 4: Bias-Variance Tradeoff & Regularization")
print("=" * 60)

X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index_col='id')
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col='id').squeeze()
X_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'), index_col='id')

SAMPLE = 30000
np.random.seed(SEED)
idx = np.random.choice(len(X_train), min(SAMPLE, len(X_train)), replace=False)
X_sub = X_train.iloc[idx]
y_sub = y_train.iloc[idx]
X_sub_s = X_scaled.iloc[idx]

def plot_learning_curve(estimator, title, X, y, ax, cv=3):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1, random_state=SEED
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation score')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('R2 Score')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    return train_mean[-1] - val_mean[-1]

# 1. Learning Curves
print("\n[1] Generating learning curves...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
gap1 = plot_learning_curve(Ridge(alpha=1.0, random_state=SEED), 'Ridge Regression', X_sub_s, y_sub, axes[0])
gap2 = plot_learning_curve(RandomForestRegressor(n_estimators=100, max_depth=15, random_state=SEED, n_jobs=-1),
                           'Random Forest', X_sub, y_sub, axes[1])
try:
    from xgboost import XGBRegressor
    gap3 = plot_learning_curve(XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1,
                                            random_state=SEED, verbosity=0, n_jobs=-1),
                               'XGBoost', X_sub, y_sub, axes[2])
except ImportError:
    gap3 = 0
    axes[2].text(0.5, 0.5, 'XGBoost not installed', transform=axes[2].transAxes, ha='center')

plt.suptitle('Learning Curves -- Bias-Variance Tradeoff', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Ridge gap (train-val):    {gap1:.6f}")
print(f"  RF gap (train-val):       {gap2:.6f}")
print(f"  XGBoost gap (train-val):  {gap3:.6f}")

# 2. Ridge alpha effect
print("\n[2] Ridge regularization (alpha effect)...")
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores_a, val_scores_a = validation_curve(
    Ridge(random_state=SEED), X_sub_s, y_sub,
    param_name='alpha', param_range=alphas, cv=3, scoring='r2', n_jobs=-1
)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
ax = axes[0]
ax.plot(alphas, train_scores_a.mean(axis=1), 'o-', label='Train', color='blue')
ax.plot(alphas, val_scores_a.mean(axis=1), 'o-', label='Validation', color='orange')
ax.set_xscale('log')
ax.set_xlabel('Alpha (Regularization)')
ax.set_ylabel('R2 Score')
ax.set_title('Ridge: Alpha vs R2', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. RF max_depth effect
print("[3] Random Forest max_depth effect...")
depths = [3, 5, 7, 10, 15, 20, 30]
train_r2, val_r2 = [], []
for d in depths:
    rf = RandomForestRegressor(n_estimators=100, max_depth=d, random_state=SEED, n_jobs=-1)
    cv_scores = cross_val_score(rf, X_sub, y_sub, cv=3, scoring='r2', n_jobs=-1)
    rf.fit(X_sub, y_sub)
    train_r2.append(rf.score(X_sub, y_sub))
    val_r2.append(cv_scores.mean())

ax = axes[1]
ax.plot(depths, train_r2, 'o-', label='Train', color='blue')
ax.plot(depths, val_r2, 'o-', label='Validation', color='orange')
ax.set_xlabel('max_depth')
ax.set_ylabel('R2 Score')
ax.set_title('Random Forest: max_depth Effect', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. XGBoost reg_lambda effect
print("[4] XGBoost reg_lambda effect...")
try:
    from xgboost import XGBRegressor
    lambdas = [0, 0.1, 1, 5, 10, 50, 100]
    train_r2_x, val_r2_x = [], []
    for l in lambdas:
        xgb = XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1,
                           reg_lambda=l, random_state=SEED, verbosity=0, n_jobs=-1)
        cv_s = cross_val_score(xgb, X_sub, y_sub, cv=3, scoring='r2', n_jobs=-1)
        xgb.fit(X_sub, y_sub)
        train_r2_x.append(xgb.score(X_sub, y_sub))
        val_r2_x.append(cv_s.mean())
    ax = axes[2]
    ax.plot(lambdas, train_r2_x, 'o-', label='Train', color='blue')
    ax.plot(lambdas, val_r2_x, 'o-', label='Validation', color='orange')
    ax.set_xlabel('reg_lambda (L2 Regularization)')
    ax.set_ylabel('R2 Score')
    ax.set_title('XGBoost: L2 Regularization Effect', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
except ImportError:
    axes[2].text(0.5, 0.5, 'XGBoost not installed', transform=axes[2].transAxes, ha='center')

plt.suptitle('Regularization & Overfitting Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'regularization_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n  Analysis:")
print("  * Ridge: Higher alpha -> more regularization -> prevents overfitting but increases bias")
print("  * Random Forest: Deeper trees -> more variance (overfitting), shallower -> more bias")
print("  * XGBoost: Higher reg_lambda -> stronger L2 penalty -> reduces overfitting")
print("\n" + "=" * 60)
print("  Bias-Variance Analysis Complete!")
print("=" * 60)

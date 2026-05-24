"""
FULL ML PIPELINE -- Optimized single-run script
=================================================
Runs Steps 3-6 in one go:
  - Step 3: Train Ridge, Random Forest, XGBoost (reduced grid for speed)
  - Step 4: Bias-Variance (learning curves on 15K sample)
  - Step 5: Evaluation metrics + plots
  - Step 6: Clustering analysis
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, precision_recall_curve,
                             silhouette_score)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data', 'processed')
RAW_DIR = os.path.join(BASE, '..', 'data')
OUTPUT_DIR = os.path.join(BASE, '..', 'outputs')
MODELS_DIR = os.path.join(BASE, '..', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

t0 = time.time()

# ================================================================
# LOAD DATA
# ================================================================
print("=" * 60)
print("  LOADING DATA")
print("=" * 60)
sys.stdout.flush()

X_train_s = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'), index_col='id')
X_test_s = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'), index_col='id')
X_train_r = pd.read_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index_col='id')
X_test_r = pd.read_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'), index_col='id')
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col='id').squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'), index_col='id').squeeze()
print(f"  Train: {X_train_s.shape}, Test: {X_test_s.shape}")
print(f"  Loaded in {time.time()-t0:.1f}s")
sys.stdout.flush()

# Subsample for CV (20K for speed)
idx = np.random.choice(len(X_train_s), 20000, replace=False)
Xs_s, Xs_r, ys = X_train_s.iloc[idx], X_train_r.iloc[idx], y_train.iloc[idx]

# ================================================================
# STEP 3: TRAIN MODELS
# ================================================================
print("\n" + "=" * 60)
print("  STEP 3: MODEL TRAINING")
print("=" * 60)
sys.stdout.flush()
results = {}

# -- Ridge --
print("\n  [Ridge] GridSearchCV...", end=" ", flush=True)
t1 = time.time()
ridge_cv = GridSearchCV(Ridge(random_state=SEED),
                        {'alpha': [0.01, 0.1, 1.0, 10.0]},
                        cv=3, scoring='r2', n_jobs=-1)
ridge_cv.fit(Xs_s, ys)
print(f"best alpha={ridge_cv.best_params_['alpha']}, CV R2={ridge_cv.best_score_:.4f} ({time.time()-t1:.1f}s)")
sys.stdout.flush()

print("  [Ridge] Training on full data...", end=" ", flush=True)
t1 = time.time()
ridge = Ridge(alpha=ridge_cv.best_params_['alpha'], random_state=SEED)
ridge.fit(X_train_s, y_train)
yp = ridge.predict(X_test_s)
results['Ridge Regression'] = {'r2': r2_score(y_test, yp), 'mae': mean_absolute_error(y_test, yp),
                                'rmse': np.sqrt(mean_squared_error(y_test, yp)), 'preds': yp}
print(f"R2={results['Ridge Regression']['r2']:.4f} ({time.time()-t1:.1f}s)")
joblib.dump(ridge, os.path.join(MODELS_DIR, 'ridge_model.pkl'))
sys.stdout.flush()

# -- Random Forest --
print("\n  [RF] GridSearchCV...", end=" ", flush=True)
t1 = time.time()
rf_cv = GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=-1),
                     {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_leaf': [5]},
                     cv=3, scoring='r2', n_jobs=-1)
rf_cv.fit(Xs_r, ys)
print(f"best={rf_cv.best_params_}, CV R2={rf_cv.best_score_:.4f} ({time.time()-t1:.1f}s)")
sys.stdout.flush()

print("  [RF] Training on full data...", end=" ", flush=True)
t1 = time.time()
rf = RandomForestRegressor(**rf_cv.best_params_, random_state=SEED, n_jobs=-1)
rf.fit(X_train_r, y_train)
yp = rf.predict(X_test_r)
results['Random Forest'] = {'r2': r2_score(y_test, yp), 'mae': mean_absolute_error(y_test, yp),
                             'rmse': np.sqrt(mean_squared_error(y_test, yp)), 'preds': yp}
print(f"R2={results['Random Forest']['r2']:.4f} ({time.time()-t1:.1f}s)")
joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
sys.stdout.flush()

# -- XGBoost --
print("\n  [XGB] GridSearchCV...", end=" ", flush=True)
t1 = time.time()
from xgboost import XGBRegressor
xgb_cv = GridSearchCV(XGBRegressor(random_state=SEED, verbosity=0, n_jobs=-1),
                      {'max_depth': [5, 7], 'learning_rate': [0.05, 0.1],
                       'n_estimators': [200, 300]},
                      cv=3, scoring='r2', n_jobs=-1)
xgb_cv.fit(Xs_r, ys)
print(f"best={xgb_cv.best_params_}, CV R2={xgb_cv.best_score_:.4f} ({time.time()-t1:.1f}s)")
sys.stdout.flush()

print("  [XGB] Training on full data...", end=" ", flush=True)
t1 = time.time()
xgb = XGBRegressor(**xgb_cv.best_params_, random_state=SEED, verbosity=0, n_jobs=-1)
xgb.fit(X_train_r, y_train)
yp = xgb.predict(X_test_r)
results['XGBoost'] = {'r2': r2_score(y_test, yp), 'mae': mean_absolute_error(y_test, yp),
                       'rmse': np.sqrt(mean_squared_error(y_test, yp)), 'preds': yp}
print(f"R2={results['XGBoost']['r2']:.4f} ({time.time()-t1:.1f}s)")
joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
sys.stdout.flush()

# Comparison
print("\n  MODEL COMPARISON:")
print(f"  {'Model':<20s} {'R2':>8s} {'MAE':>8s} {'RMSE':>8s}")
print(f"  {'-'*46}")
for n, m in results.items():
    print(f"  {n:<20s} {m['r2']:>8.4f} {m['mae']:>8.4f} {m['rmse']:>8.4f}")
sys.stdout.flush()

# Feature importance plot
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
pd.Series(np.abs(ridge.coef_), index=FEATURES).sort_values().plot.barh(ax=axes[0], color='#2196F3')
axes[0].set_title('Ridge |Coefficients|', fontweight='bold')
pd.Series(rf.feature_importances_, index=FEATURES).sort_values().plot.barh(ax=axes[1], color='#4CAF50')
axes[1].set_title('Random Forest Importance', fontweight='bold')
pd.Series(xgb.feature_importances_, index=FEATURES).sort_values().plot.barh(ax=axes[2], color='#FF5722')
axes[2].set_title('XGBoost Importance', fontweight='bold')
plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# Bar comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
names = list(results.keys())
colors = ['#2196F3', '#4CAF50', '#FF5722']
for i, (metric, title) in enumerate([('r2','R2 Score'),('mae','MAE'),('rmse','RMSE')]):
    vals = [results[n][metric] for n in names]
    axes[i].bar(names, vals, color=colors, edgecolor='white', linewidth=2)
    axes[i].set_title(title, fontweight='bold')
    for j, v in enumerate(vals):
        axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ================================================================
# STEP 4: BIAS-VARIANCE (on 15K sample for speed)
# ================================================================
print("\n" + "=" * 60)
print("  STEP 4: BIAS-VARIANCE ANALYSIS")
print("=" * 60)
sys.stdout.flush()

bv_idx = np.random.choice(len(X_train_s), 15000, replace=False)
Xbv_s, Xbv_r, ybv = X_train_s.iloc[bv_idx], X_train_r.iloc[bv_idx], y_train.iloc[bv_idx]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for i, (est, title, X_d) in enumerate([
    (Ridge(alpha=1.0, random_state=SEED), 'Ridge', Xbv_s),
    (RandomForestRegressor(n_estimators=100, max_depth=15, random_state=SEED, n_jobs=-1), 'Random Forest', Xbv_r),
    (XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=SEED, verbosity=0, n_jobs=-1), 'XGBoost', Xbv_r),
]):
    print(f"  Learning curve: {title}...", end=" ", flush=True)
    t1 = time.time()
    sizes, tr_sc, va_sc = learning_curve(est, X_d, ybv, cv=3, scoring='r2',
                                          train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1, random_state=SEED)
    axes[i].fill_between(sizes, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.1, color='blue')
    axes[i].fill_between(sizes, va_sc.mean(1)-va_sc.std(1), va_sc.mean(1)+va_sc.std(1), alpha=0.1, color='orange')
    axes[i].plot(sizes, tr_sc.mean(1), 'o-', color='blue', label='Train')
    axes[i].plot(sizes, va_sc.mean(1), 'o-', color='orange', label='Validation')
    axes[i].set_title(title, fontweight='bold')
    axes[i].set_xlabel('Training Size')
    axes[i].set_ylabel('R2')
    axes[i].legend(loc='lower right')
    axes[i].grid(True, alpha=0.3)
    gap = tr_sc.mean(1)[-1] - va_sc.mean(1)[-1]
    print(f"gap={gap:.4f} ({time.time()-t1:.1f}s)")
    sys.stdout.flush()
plt.suptitle('Learning Curves -- Bias-Variance Tradeoff', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Regularization plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
print("  Regularization: Ridge alpha...", end=" ", flush=True)
from sklearn.model_selection import validation_curve
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
tr_a, va_a = validation_curve(Ridge(random_state=SEED), Xbv_s, ybv, param_name='alpha',
                               param_range=alphas, cv=3, scoring='r2', n_jobs=-1)
axes[0].plot(alphas, tr_a.mean(1), 'o-', color='blue', label='Train')
axes[0].plot(alphas, va_a.mean(1), 'o-', color='orange', label='Val')
axes[0].set_xscale('log'); axes[0].set_title('Ridge: Alpha', fontweight='bold')
axes[0].set_xlabel('alpha'); axes[0].set_ylabel('R2'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
print("done", flush=True)

print("  Regularization: RF max_depth...", end=" ", flush=True)
depths = [3, 5, 10, 15, 20]
tr_d, va_d = [], []
for d in depths:
    r = RandomForestRegressor(n_estimators=50, max_depth=d, random_state=SEED, n_jobs=-1)
    cv = cross_val_score(r, Xbv_r, ybv, cv=3, scoring='r2', n_jobs=-1)
    r.fit(Xbv_r, ybv)
    tr_d.append(r.score(Xbv_r, ybv)); va_d.append(cv.mean())
axes[1].plot(depths, tr_d, 'o-', color='blue', label='Train')
axes[1].plot(depths, va_d, 'o-', color='orange', label='Val')
axes[1].set_title('RF: max_depth', fontweight='bold')
axes[1].set_xlabel('max_depth'); axes[1].set_ylabel('R2'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
print("done", flush=True)

print("  Regularization: XGB reg_lambda...", end=" ", flush=True)
lambdas = [0, 1, 5, 10, 50]
tr_l, va_l = [], []
for l in lambdas:
    x = XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, reg_lambda=l,
                     random_state=SEED, verbosity=0, n_jobs=-1)
    cv = cross_val_score(x, Xbv_r, ybv, cv=3, scoring='r2', n_jobs=-1)
    x.fit(Xbv_r, ybv)
    tr_l.append(x.score(Xbv_r, ybv)); va_l.append(cv.mean())
axes[2].plot(lambdas, tr_l, 'o-', color='blue', label='Train')
axes[2].plot(lambdas, va_l, 'o-', color='orange', label='Val')
axes[2].set_title('XGBoost: reg_lambda', fontweight='bold')
axes[2].set_xlabel('reg_lambda'); axes[2].set_ylabel('R2'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
print("done", flush=True)

plt.suptitle('Regularization Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'regularization_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# ================================================================
# STEP 5: EVALUATION
# ================================================================
print("\n" + "=" * 60)
print("  STEP 5: MODEL EVALUATION")
print("=" * 60)
sys.stdout.flush()

THRESHOLD = 0.5
y_bin = (y_test >= THRESHOLD).astype(int)
all_metrics = {}

for name, m_data in results.items():
    yp = m_data['preds']
    yp_b = (yp >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, yp_b).ravel()
    fpr, tpr, _ = roc_curve(y_bin, yp)
    roc_auc = auc(fpr, tpr)
    all_metrics[name] = {
        'r2': m_data['r2'], 'mae': m_data['mae'], 'rmse': m_data['rmse'],
        'accuracy': accuracy_score(y_bin, yp_b),
        'precision': precision_score(y_bin, yp_b, zero_division=0),
        'recall': recall_score(y_bin, yp_b, zero_division=0),
        'specificity': tn/(tn+fp) if (tn+fp) > 0 else 0,
        'f1': f1_score(y_bin, yp_b, zero_division=0),
        'auc': roc_auc, 'fpr': fpr, 'tpr': tpr,
    }
    print(f"  {name}: R2={m_data['r2']:.4f} Acc={all_metrics[name]['accuracy']:.4f} "
          f"F1={all_metrics[name]['f1']:.4f} AUC={roc_auc:.4f}")
sys.stdout.flush()

# Save metrics JSON for dashboard
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

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cmaps = ['Blues', 'Greens', 'Oranges']
for i, (name, m) in enumerate(results.items()):
    cm = confusion_matrix(y_bin, (m['preds'] >= THRESHOLD).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[i], ax=axes[i],
                xticklabels=['Safe','At-Risk'], yticklabels=['Safe','At-Risk'])
    axes[i].set_title(f'{name}', fontweight='bold')
    axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
plt.suptitle(f'Confusion Matrices (Threshold={THRESHOLD})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

# ROC + PR curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
clrs = ['#2196F3', '#4CAF50', '#FF5722']
for i, (name, m) in enumerate(all_metrics.items()):
    axes[0].plot(m['fpr'], m['tpr'], color=clrs[i], lw=2, label=f"{name} (AUC={m['auc']:.4f})")
axes[0].plot([0,1],[0,1],'k--',lw=1)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('ROC Curves', fontweight='bold'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

for i, (name, m) in enumerate(results.items()):
    prec, rec, _ = precision_recall_curve(y_bin, m['preds'])
    axes[1].plot(rec, prec, color=clrs[i], lw=2, label=name)
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves', fontweight='bold'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_pr_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Threshold analysis (best model)
best_name = max(all_metrics, key=lambda x: all_metrics[x]['auc'])
best_preds = results[best_name]['preds']
thresholds = np.arange(0.35, 0.65, 0.02)
fig, ax = plt.subplots(figsize=(10, 6))
for metric_name, style, color in [('precision','o-','#2196F3'),('recall','s-','#4CAF50'),
                                    ('f1','^-','#FF5722'),('accuracy','D-','#9C27B0')]:
    vals = []
    for t in thresholds:
        yb = (best_preds >= t).astype(int)
        yt = (y_test >= t).astype(int)
        if metric_name == 'precision': vals.append(precision_score(yt, yb, zero_division=0))
        elif metric_name == 'recall': vals.append(recall_score(yt, yb, zero_division=0))
        elif metric_name == 'f1': vals.append(f1_score(yt, yb, zero_division=0))
        else: vals.append(accuracy_score(yt, yb))
    ax.plot(thresholds, vals, style, label=metric_name.title(), color=color)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
ax.set_title(f'Threshold Analysis -- {best_name}', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Plots saved.")
sys.stdout.flush()

# ================================================================
# STEP 6: CLUSTERING
# ================================================================
print("\n" + "=" * 60)
print("  STEP 6: CLUSTERING ANALYSIS")
print("=" * 60)
sys.stdout.flush()

train_full = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'), index_col='id')
csample = train_full.sample(n=30000, random_state=SEED)
Xc = csample[FEATURES]
yc = csample['FloodProbability']
sc = StandardScaler()
Xc_s = sc.fit_transform(Xc)

# Elbow + Silhouette
print("  Finding optimal K...", flush=True)
K_range = range(2, 8)
inertias, sils = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    lb = km.fit_predict(Xc_s)
    inertias.append(km.inertia_)
    s = silhouette_score(Xc_s, lb, sample_size=5000)
    sils.append(s)
    print(f"    K={k}: Silhouette={s:.4f}")
sys.stdout.flush()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-', lw=2, ms=8)
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia'); axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[1].plot(K_range, sils, 'ro-', lw=2, ms=8)
axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette'); axes[1].set_title('Silhouette Analysis', fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_elbow_silhouette.png'), dpi=150, bbox_inches='tight')
plt.close()

# K=4 clustering
OPTIMAL_K = 4
km4 = KMeans(n_clusters=OPTIMAL_K, random_state=SEED, n_init=10)
clusters = km4.fit_predict(Xc_s)
csample['Cluster'] = clusters
cluster_names = ['Low-Risk Rural', 'Moderate Urban', 'High-Risk Floodplain', 'Critical Zone']

# PCA
pca = PCA(n_components=2, random_state=SEED)
Xpca = pca.fit_transform(Xc_s)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
clr4 = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
for i in range(OPTIMAL_K):
    mask = clusters == i
    axes[0].scatter(Xpca[mask, 0], Xpca[mask, 1], c=clr4[i], alpha=0.3, s=10,
                    label=f'C{i}: {cluster_names[i]}')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title('K-Means Clusters (PCA)', fontweight='bold'); axes[0].legend(fontsize=8)
sc2 = axes[1].scatter(Xpca[:, 0], Xpca[:, 1], c=yc, cmap='RdYlGn_r', alpha=0.3, s=10)
plt.colorbar(sc2, ax=axes[1], label='FloodProbability')
axes[1].set_title('PCA by Flood Probability', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_pca.png'), dpi=150, bbox_inches='tight')
plt.close()

# Cluster profiles
profile = csample.groupby('Cluster')[FEATURES + ['FloodProbability']].mean()
fig, ax = plt.subplots(figsize=(16, 6))
pn = (profile - profile.min()) / (profile.max() - profile.min())
sns.heatmap(pn.T, annot=profile.T.round(1), fmt='', cmap='YlOrRd',
            xticklabels=[f'C{i}: {cluster_names[i]}' for i in range(OPTIMAL_K)], ax=ax, linewidths=0.5)
ax.set_title('Cluster Profiles', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()

# DBSCAN
db = DBSCAN(eps=3.5, min_samples=10).fit_predict(Xc_s[:10000])
ndb = len(set(db)) - (1 if -1 in db else 0)
noise = (db == -1).sum()
print(f"  DBSCAN: {ndb} clusters, {noise} noise ({noise/10000*100:.1f}%)")

# Save cluster JSON
cdata = {
    'n_clusters': OPTIMAL_K, 'cluster_names': cluster_names,
    'cluster_sizes': [int((clusters == i).sum()) for i in range(OPTIMAL_K)],
    'mean_flood_probability': [float(csample.loc[clusters == i, 'FloodProbability'].mean()) for i in range(OPTIMAL_K)]
}
with open(os.path.join(OUTPUT_DIR, 'cluster_profiles.json'), 'w') as f:
    json.dump(cdata, f, indent=2)

for c in range(OPTIMAL_K):
    fp = csample.loc[clusters == c, 'FloodProbability']
    print(f"  Cluster {c} ({cluster_names[c]}): n={int((clusters==c).sum())}, mean_FP={fp.mean():.4f}")
sys.stdout.flush()

# ================================================================
# DONE
# ================================================================
total = time.time() - t0
print("\n" + "=" * 60)
print(f"  ALL STEPS COMPLETE! Total time: {total:.0f}s ({total/60:.1f} min)")
print("=" * 60)
print(f"  Models saved to: {MODELS_DIR}")
print(f"  Plots saved to:  {OUTPUT_DIR}")
print(f"  Metrics saved to: {os.path.join(OUTPUT_DIR, 'model_metrics.json')}")

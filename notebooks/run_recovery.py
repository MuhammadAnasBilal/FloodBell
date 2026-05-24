"""
Recovery script: Train XGBoost + run Steps 4-6
Memory-safe: n_jobs=2, smaller samples
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
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, precision_recall_curve,
                             silhouette_score)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data', 'processed')
RAW_DIR = os.path.join(BASE, '..', 'data')
OUTPUT_DIR = os.path.join(BASE, '..', 'outputs')
MODELS_DIR = os.path.join(BASE, '..', 'models')

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

# Load data
print("Loading data...", flush=True)
X_train_s = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'), index_col='id')
X_test_s = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'), index_col='id')
X_train_r = pd.read_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index_col='id')
X_test_r = pd.read_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'), index_col='id')
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col='id').squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'), index_col='id').squeeze()
print(f"  Loaded. Train={X_train_r.shape}", flush=True)

# Load existing models
ridge = joblib.load(os.path.join(MODELS_DIR, 'ridge_model.pkl'))
rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
print("  Ridge and RF loaded from disk.", flush=True)

# ================================================================
# TRAIN XGBOOST (memory-safe: n_jobs=2, no GridSearchCV parallel)
# ================================================================
print("\n=== TRAINING XGBOOST (memory-safe) ===", flush=True)

# Quick manual search on 15K sample
idx = np.random.choice(len(X_train_r), 15000, replace=False)
Xs, ys = X_train_r.iloc[idx], y_train.iloc[idx]

best_score, best_params = -1, {}
configs = [
    {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200},
    {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 200},
    {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 9, 'learning_rate': 0.1, 'n_estimators': 200},
]
for cfg in configs:
    xgb_test = XGBRegressor(**cfg, random_state=SEED, verbosity=0, n_jobs=2)
    cv = cross_val_score(xgb_test, Xs, ys, cv=3, scoring='r2', n_jobs=1)
    score = cv.mean()
    print(f"  {cfg} -> CV R2={score:.4f}", flush=True)
    if score > best_score:
        best_score, best_params = score, cfg

print(f"  Best: {best_params}, R2={best_score:.4f}", flush=True)

# Train on full data with n_jobs=2
print("  Training XGBoost on full data (n_jobs=2)...", flush=True)
t1 = time.time()
xgb = XGBRegressor(**best_params, random_state=SEED, verbosity=0, n_jobs=2)
xgb.fit(X_train_r, y_train)
yp_xgb = xgb.predict(X_test_r)
r2_xgb = r2_score(y_test, yp_xgb)
print(f"  XGBoost R2={r2_xgb:.4f} ({time.time()-t1:.0f}s)", flush=True)
joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))

# Collect all predictions
yp_ridge = ridge.predict(X_test_s)
yp_rf = rf.predict(X_test_r)
results = {
    'Ridge Regression': {'preds': yp_ridge, 'r2': r2_score(y_test, yp_ridge),
                          'mae': mean_absolute_error(y_test, yp_ridge),
                          'rmse': np.sqrt(mean_squared_error(y_test, yp_ridge))},
    'Random Forest': {'preds': yp_rf, 'r2': r2_score(y_test, yp_rf),
                       'mae': mean_absolute_error(y_test, yp_rf),
                       'rmse': np.sqrt(mean_squared_error(y_test, yp_rf))},
    'XGBoost': {'preds': yp_xgb, 'r2': r2_xgb,
                 'mae': mean_absolute_error(y_test, yp_xgb),
                 'rmse': np.sqrt(mean_squared_error(y_test, yp_xgb))},
}

print("\n  MODEL COMPARISON:", flush=True)
for n, m in results.items():
    print(f"    {n:<20s} R2={m['r2']:.4f} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}", flush=True)

# Feature importance
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
pd.Series(np.abs(ridge.coef_), index=FEATURES).sort_values().plot.barh(ax=axes[0], color='#2196F3')
axes[0].set_title('Ridge |Coefficients|', fontweight='bold')
pd.Series(rf.feature_importances_, index=FEATURES).sort_values().plot.barh(ax=axes[1], color='#4CAF50')
axes[1].set_title('Random Forest', fontweight='bold')
pd.Series(xgb.feature_importances_, index=FEATURES).sort_values().plot.barh(ax=axes[2], color='#FF5722')
axes[2].set_title('XGBoost', fontweight='bold')
plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
names = list(results.keys())
colors = ['#2196F3', '#4CAF50', '#FF5722']
for i, (metric, title) in enumerate([('r2','R2'),('mae','MAE'),('rmse','RMSE')]):
    vals = [results[n][metric] for n in names]
    axes[i].bar(names, vals, color=colors)
    axes[i].set_title(title, fontweight='bold')
    for j, v in enumerate(vals): axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ================================================================
# STEP 5: EVALUATION
# ================================================================
print("\n=== STEP 5: EVALUATION ===", flush=True)
THRESHOLD = 0.5
y_bin = (y_test >= THRESHOLD).astype(int)
all_metrics = {}

for name, m in results.items():
    yp = m['preds']
    yp_b = (yp >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, yp_b).ravel()
    fpr, tpr, _ = roc_curve(y_bin, yp)
    roc_auc = auc(fpr, tpr)
    all_metrics[name] = {
        'r2': m['r2'], 'mae': m['mae'], 'rmse': m['rmse'],
        'accuracy': accuracy_score(y_bin, yp_b),
        'precision': precision_score(y_bin, yp_b, zero_division=0),
        'recall': recall_score(y_bin, yp_b, zero_division=0),
        'specificity': tn/(tn+fp) if (tn+fp) > 0 else 0,
        'f1': f1_score(y_bin, yp_b, zero_division=0),
        'auc': roc_auc, 'fpr': fpr, 'tpr': tpr,
    }
    print(f"  {name}: Acc={all_metrics[name]['accuracy']:.4f} F1={all_metrics[name]['f1']:.4f} AUC={roc_auc:.4f}", flush=True)

# Save metrics JSON
ms = {k: [all_metrics[n][k] for n in all_metrics] for k in ['r2','mae','rmse','accuracy','precision','recall','f1','auc']}
ms['models'] = list(all_metrics.keys())
with open(os.path.join(OUTPUT_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(ms, f, indent=2)

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, m) in enumerate(results.items()):
    cm = confusion_matrix(y_bin, (m['preds'] >= THRESHOLD).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap=['Blues','Greens','Oranges'][i], ax=axes[i],
                xticklabels=['Safe','At-Risk'], yticklabels=['Safe','At-Risk'])
    axes[i].set_title(name, fontweight='bold')
    axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

# ROC + PR
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
clrs = ['#2196F3', '#4CAF50', '#FF5722']
for i, (name, m) in enumerate(all_metrics.items()):
    axes[0].plot(m['fpr'], m['tpr'], color=clrs[i], lw=2, label=f"{name} (AUC={m['auc']:.4f})")
axes[0].plot([0,1],[0,1],'k--'); axes[0].set_title('ROC Curves', fontweight='bold')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
for i, (name, m) in enumerate(results.items()):
    pr, rc, _ = precision_recall_curve(y_bin, m['preds'])
    axes[1].plot(rc, pr, color=clrs[i], lw=2, label=name)
axes[1].set_title('Precision-Recall', fontweight='bold')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_pr_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Threshold analysis
best_name = max(all_metrics, key=lambda x: all_metrics[x]['auc'])
bp = results[best_name]['preds']
thresholds = np.arange(0.35, 0.65, 0.02)
fig, ax = plt.subplots(figsize=(10, 6))
for mn, st, cl in [('precision','o-','#2196F3'),('recall','s-','#4CAF50'),('f1','^-','#FF5722'),('accuracy','D-','#9C27B0')]:
    vals = []
    for t in thresholds:
        yb, yt2 = (bp >= t).astype(int), (y_test >= t).astype(int)
        if mn == 'precision': vals.append(precision_score(yt2, yb, zero_division=0))
        elif mn == 'recall': vals.append(recall_score(yt2, yb, zero_division=0))
        elif mn == 'f1': vals.append(f1_score(yt2, yb, zero_division=0))
        else: vals.append(accuracy_score(yt2, yb))
    ax.plot(thresholds, vals, st, label=mn.title(), color=cl)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Threshold'); ax.set_ylabel('Score'); ax.set_title(f'Threshold Analysis -- {best_name}', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Evaluation plots saved.", flush=True)

# Free memory
del X_train_s, X_test_s, X_train_r, X_test_r, y_train, y_test
import gc; gc.collect()

# ================================================================
# STEP 4: BIAS-VARIANCE (10K sample, n_jobs=1)
# ================================================================
print("\n=== STEP 4: BIAS-VARIANCE ===", flush=True)
X_tr_r = pd.read_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index_col='id')
X_tr_s = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'), index_col='id')
y_tr = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col='id').squeeze()
bidx = np.random.choice(len(X_tr_r), 10000, replace=False)
Xb_r, Xb_s, yb = X_tr_r.iloc[bidx], X_tr_s.iloc[bidx], y_tr.iloc[bidx]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for i, (est, title, Xd) in enumerate([
    (Ridge(alpha=0.1, random_state=SEED), 'Ridge', Xb_s),
    (RandomForestRegressor(n_estimators=50, max_depth=15, random_state=SEED, n_jobs=2), 'Random Forest', Xb_r),
    (XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=SEED, verbosity=0, n_jobs=2), 'XGBoost', Xb_r),
]):
    print(f"  Learning curve: {title}...", end=" ", flush=True)
    t1 = time.time()
    sizes, tr_sc, va_sc = learning_curve(est, Xd, yb, cv=3, scoring='r2',
                                          train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1, random_state=SEED)
    axes[i].fill_between(sizes, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.1, color='blue')
    axes[i].fill_between(sizes, va_sc.mean(1)-va_sc.std(1), va_sc.mean(1)+va_sc.std(1), alpha=0.1, color='orange')
    axes[i].plot(sizes, tr_sc.mean(1), 'o-', color='blue', label='Train')
    axes[i].plot(sizes, va_sc.mean(1), 'o-', color='orange', label='Validation')
    axes[i].set_title(title, fontweight='bold'); axes[i].set_xlabel('Size'); axes[i].set_ylabel('R2')
    axes[i].legend(loc='lower right'); axes[i].grid(True, alpha=0.3)
    print(f"done ({time.time()-t1:.0f}s)", flush=True)
plt.suptitle('Learning Curves -- Bias-Variance Tradeoff', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Regularization
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
from sklearn.model_selection import validation_curve
print("  Regularization plots...", flush=True)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
tr_a, va_a = validation_curve(Ridge(random_state=SEED), Xb_s, yb, param_name='alpha',
                               param_range=alphas, cv=3, scoring='r2', n_jobs=1)
axes[0].plot(alphas, tr_a.mean(1), 'o-', color='blue', label='Train')
axes[0].plot(alphas, va_a.mean(1), 'o-', color='orange', label='Val')
axes[0].set_xscale('log'); axes[0].set_title('Ridge: Alpha', fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

depths = [3, 5, 10, 15, 20]
tr_d, va_d = [], []
for d in depths:
    r = RandomForestRegressor(n_estimators=30, max_depth=d, random_state=SEED, n_jobs=2)
    cv = cross_val_score(r, Xb_r, yb, cv=3, scoring='r2', n_jobs=1)
    r.fit(Xb_r, yb); tr_d.append(r.score(Xb_r, yb)); va_d.append(cv.mean())
axes[1].plot(depths, tr_d, 'o-', color='blue', label='Train')
axes[1].plot(depths, va_d, 'o-', color='orange', label='Val')
axes[1].set_title('RF: max_depth', fontweight='bold'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

lambdas = [0, 1, 5, 10, 50]
tr_l, va_l = [], []
for l in lambdas:
    x = XGBRegressor(n_estimators=100, max_depth=7, reg_lambda=l, random_state=SEED, verbosity=0, n_jobs=2)
    cv = cross_val_score(x, Xb_r, yb, cv=3, scoring='r2', n_jobs=1)
    x.fit(Xb_r, yb); tr_l.append(x.score(Xb_r, yb)); va_l.append(cv.mean())
axes[2].plot(lambdas, tr_l, 'o-', color='blue', label='Train')
axes[2].plot(lambdas, va_l, 'o-', color='orange', label='Val')
axes[2].set_title('XGBoost: reg_lambda', fontweight='bold'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
plt.suptitle('Regularization Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'regularization_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Bias-variance plots saved.", flush=True)

del X_tr_r, X_tr_s, y_tr; gc.collect()

# ================================================================
# STEP 6: CLUSTERING (20K sample)
# ================================================================
print("\n=== STEP 6: CLUSTERING ===", flush=True)
train_full = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'), index_col='id')
cs = train_full.sample(n=20000, random_state=SEED)
del train_full; gc.collect()
Xc = cs[FEATURES]; yc = cs['FloodProbability']
sc = StandardScaler(); Xc_s = sc.fit_transform(Xc)

K_range = range(2, 8)
inertias, sils = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    lb = km.fit_predict(Xc_s)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xc_s, lb, sample_size=5000))
    print(f"  K={k}: Sil={sils[-1]:.4f}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-'); axes[0].set_title('Elbow Method', fontweight='bold'); axes[0].grid(True, alpha=0.3)
axes[1].plot(K_range, sils, 'ro-'); axes[1].set_title('Silhouette', fontweight='bold'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_elbow_silhouette.png'), dpi=150, bbox_inches='tight')
plt.close()

OPTIMAL_K = 4
clusters = KMeans(n_clusters=OPTIMAL_K, random_state=SEED, n_init=10).fit_predict(Xc_s)
cs['Cluster'] = clusters
cluster_names = ['Low-Risk Rural', 'Moderate Urban', 'High-Risk Floodplain', 'Critical Zone']

pca = PCA(n_components=2, random_state=SEED); Xpca = pca.fit_transform(Xc_s)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
clr4 = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
for i in range(OPTIMAL_K):
    mask = clusters == i
    axes[0].scatter(Xpca[mask,0], Xpca[mask,1], c=clr4[i], alpha=0.3, s=10, label=f'C{i}: {cluster_names[i]}')
axes[0].set_title('K-Means (PCA)', fontweight='bold'); axes[0].legend(fontsize=8)
sc2 = axes[1].scatter(Xpca[:,0], Xpca[:,1], c=yc, cmap='RdYlGn_r', alpha=0.3, s=10)
plt.colorbar(sc2, ax=axes[1]); axes[1].set_title('PCA by FloodProbability', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_pca.png'), dpi=150, bbox_inches='tight')
plt.close()

profile = cs.groupby('Cluster')[FEATURES + ['FloodProbability']].mean()
fig, ax = plt.subplots(figsize=(16, 6))
pn = (profile - profile.min()) / (profile.max() - profile.min())
sns.heatmap(pn.T, annot=profile.T.round(1), fmt='', cmap='YlOrRd',
            xticklabels=[f'C{i}: {cluster_names[i]}' for i in range(OPTIMAL_K)], ax=ax)
ax.set_title('Cluster Profiles', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()

db = DBSCAN(eps=3.5, min_samples=10).fit_predict(Xc_s[:8000])
print(f"  DBSCAN: {len(set(db))-(1 if -1 in db else 0)} clusters", flush=True)

cdata = {
    'n_clusters': OPTIMAL_K, 'cluster_names': cluster_names,
    'cluster_sizes': [int((clusters==i).sum()) for i in range(OPTIMAL_K)],
    'mean_flood_probability': [float(cs.loc[clusters==i,'FloodProbability'].mean()) for i in range(OPTIMAL_K)]
}
with open(os.path.join(OUTPUT_DIR, 'cluster_profiles.json'), 'w') as f:
    json.dump(cdata, f, indent=2)

for c in range(OPTIMAL_K):
    fp = cs.loc[clusters==c, 'FloodProbability']
    print(f"  Cluster {c} ({cluster_names[c]}): n={int((clusters==c).sum())}, mean_FP={fp.mean():.4f}", flush=True)

total = time.time() - t0
print(f"\n{'='*60}")
print(f"  ALL COMPLETE! Total: {total:.0f}s ({total/60:.1f} min)")
print(f"  Models: {os.listdir(MODELS_DIR)}")
print(f"  Outputs: {os.listdir(OUTPUT_DIR)}")
print(f"{'='*60}")

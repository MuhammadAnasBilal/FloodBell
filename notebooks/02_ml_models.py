"""
Step 3: Machine Learning Models -- Baseline, Intermediate, Advanced
====================================================================
Three models with structured comparison:
  1. Baseline:      Ridge Regression
  2. Intermediate:  Random Forest Regressor
  3. Advanced:      XGBoost Regressor

Each model: hyperparameter tuning (GridSearchCV on 50K sample),
train on full data, feature importance, save to disk.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings('ignore')

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
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

print("=" * 60)
print("  STEP 3: Machine Learning Models")
print("=" * 60)

# Load Data
print("\n[1] Loading preprocessed data...")
X_train_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'), index_col='id')
X_test_scaled = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'), index_col='id')
X_train_raw = pd.read_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index_col='id')
X_test_raw = pd.read_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'), index_col='id')
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col='id').squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'), index_col='id').squeeze()
print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# Subsample for hyperparameter tuning
SAMPLE_SIZE = 50000
np.random.seed(SEED)
sample_idx = np.random.choice(len(X_train_scaled), size=min(SAMPLE_SIZE, len(X_train_scaled)), replace=False)
X_sample_scaled = X_train_scaled.iloc[sample_idx]
X_sample_raw = X_train_raw.iloc[sample_idx]
y_sample = y_train.iloc[sample_idx]

results = {}

# == Model 1: Ridge Regression (Baseline) ==
print("\n" + "-" * 60)
print("  Model 1: Ridge Regression (Baseline)")
print("-" * 60)

ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_cv = GridSearchCV(Ridge(random_state=SEED), ridge_params, cv=5, scoring='r2', n_jobs=-1)
ridge_cv.fit(X_sample_scaled, y_sample)
print(f"  Best alpha: {ridge_cv.best_params_['alpha']}")
print(f"  Best CV R2: {ridge_cv.best_score_:.6f}")

ridge_model = Ridge(alpha=ridge_cv.best_params_['alpha'], random_state=SEED)
ridge_model.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_model.predict(X_test_scaled)
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"  Test R2:    {r2_ridge:.6f}")
print(f"  Test MAE:   {mae_ridge:.6f}")
print(f"  Test RMSE:  {rmse_ridge:.6f}")
results['Ridge Regression'] = {'r2': r2_ridge, 'mae': mae_ridge, 'rmse': rmse_ridge}
joblib.dump(ridge_model, os.path.join(MODELS_DIR, 'ridge_model.pkl'))

# == Model 2: Random Forest (Intermediate) ==
print("\n" + "-" * 60)
print("  Model 2: Random Forest Regressor (Intermediate)")
print("-" * 60)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_leaf': [5, 10]
}
rf_cv = GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=-1), rf_params, cv=3, scoring='r2', n_jobs=-1)
rf_cv.fit(X_sample_raw, y_sample)
print(f"  Best params: {rf_cv.best_params_}")
print(f"  Best CV R2:  {rf_cv.best_score_:.6f}")

rf_model = RandomForestRegressor(**rf_cv.best_params_, random_state=SEED, n_jobs=-1)
rf_model.fit(X_train_raw, y_train)

y_pred_rf = rf_model.predict(X_test_raw)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"  Test R2:    {r2_rf:.6f}")
print(f"  Test MAE:   {mae_rf:.6f}")
print(f"  Test RMSE:  {rmse_rf:.6f}")
results['Random Forest'] = {'r2': r2_rf, 'mae': mae_rf, 'rmse': rmse_rf}
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_model.pkl'))

# == Model 3: XGBoost (Advanced) ==
print("\n" + "-" * 60)
print("  Model 3: XGBoost Regressor (Advanced)")
print("-" * 60)

xgb_model = None
try:
    from xgboost import XGBRegressor
    xgb_params = {
        'max_depth': [5, 7, 9],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [200, 300],
        'reg_lambda': [1, 5]
    }
    xgb_cv = GridSearchCV(XGBRegressor(random_state=SEED, verbosity=0, n_jobs=-1),
                          xgb_params, cv=3, scoring='r2', n_jobs=-1)
    xgb_cv.fit(X_sample_raw, y_sample)
    print(f"  Best params: {xgb_cv.best_params_}")
    print(f"  Best CV R2:  {xgb_cv.best_score_:.6f}")

    xgb_model = XGBRegressor(**xgb_cv.best_params_, random_state=SEED, verbosity=0, n_jobs=-1)
    xgb_model.fit(X_train_raw, y_train)

    y_pred_xgb = xgb_model.predict(X_test_raw)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    print(f"  Test R2:    {r2_xgb:.6f}")
    print(f"  Test MAE:   {mae_xgb:.6f}")
    print(f"  Test RMSE:  {rmse_xgb:.6f}")
    results['XGBoost'] = {'r2': r2_xgb, 'mae': mae_xgb, 'rmse': rmse_xgb}
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
except ImportError:
    print("  XGBoost not installed. Skipping.")

# == Comparison Table ==
print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)
print(f"  {'Model':<20s} {'R2':>10s} {'MAE':>10s} {'RMSE':>10s}")
print(f"  {'-' * 52}")
for name, m in results.items():
    print(f"  {name:<20s} {m['r2']:>10.6f} {m['mae']:>10.6f} {m['rmse']:>10.6f}")

# Save metrics
metrics = {
    'models': list(results.keys()),
    'r2': [m['r2'] for m in results.values()],
    'mae': [m['mae'] for m in results.values()],
    'rmse': [m['rmse'] for m in results.values()],
}
with open(os.path.join(OUTPUT_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

# == Feature Importance Plots ==
print("\n[5] Plotting feature importances...")
fig, axes = plt.subplots(1, 3 if xgb_model else 2, figsize=(20, 8))

ax = axes[0]
coefs = pd.Series(np.abs(ridge_model.coef_), index=FEATURES).sort_values()
coefs.plot.barh(ax=ax, color='#2196F3')
ax.set_title('Ridge Regression\n(|Coefficients|)', fontweight='bold')
ax.set_xlabel('Absolute Coefficient Value')

ax = axes[1]
rf_imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
rf_imp.plot.barh(ax=ax, color='#4CAF50')
ax.set_title('Random Forest\n(Feature Importance)', fontweight='bold')
ax.set_xlabel('Importance')

if xgb_model:
    ax = axes[2]
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values()
    xgb_imp.plot.barh(ax=ax, color='#FF5722')
    ax.set_title('XGBoost\n(Feature Importance)', fontweight='bold')
    ax.set_xlabel('Importance')

plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
model_names = list(results.keys())
colors = ['#2196F3', '#4CAF50', '#FF5722'][:len(model_names)]

for i, (metric, title) in enumerate([('r2', 'R2 Score (higher is better)'),
                                       ('mae', 'MAE (lower is better)'),
                                       ('rmse', 'RMSE (lower is better)')]):
    vals = [results[m][metric] for m in model_names]
    axes[i].bar(model_names, vals, color=colors, edgecolor='white', linewidth=2)
    axes[i].set_title(title, fontweight='bold')
    axes[i].set_ylabel(metric.upper())
    for j, v in enumerate(vals):
        axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n  All models saved to models/")
print("  Plots saved to outputs/")
print("\n" + "=" * 60)
print("  ML Model Training Complete!")
print("=" * 60)

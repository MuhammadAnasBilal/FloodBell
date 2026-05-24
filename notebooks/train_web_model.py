"""
Optimized, Memory-Safe Training Script for Web App
===================================================
Trains a lightweight XGBoost model that is small enough for web deployment
and won't crash your computer's memory.
"""

import os, time, json
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor
import joblib

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data', 'processed')
MODELS_DIR = os.path.join(BASE, '..', 'models')
OUTPUT_DIR = os.path.join(BASE, '..', 'outputs')

print("="*50)
print("  TRAINING OPTIMIZED WEB MODEL")
print("="*50)

# 1. Load Data
print("\n1. Loading Data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index_col='id')
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col='id').squeeze()
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'), index_col='id')
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'), index_col='id').squeeze()

# 2. Train Lightweight XGBoost
# max_depth=5 and n_estimators=100 keeps the model size very small (a few MBs)
# n_jobs=2 prevents the computer from running out of memory
print("\n2. Training Lightweight XGBoost (this will take 1-2 minutes)...")
t0 = time.time()
xgb = XGBRegressor(max_depth=5, n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=2)
xgb.fit(X_train, y_train)
print(f"   Done in {time.time() - t0:.1f} seconds.")

# 3. Evaluate
print("\n3. Evaluating Model...")
y_pred = xgb.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Convert to binary classification for metrics (Threshold = 0.5)
y_pred_bin = (y_pred >= 0.5).astype(int)
y_test_bin = (y_test >= 0.5).astype(int)

metrics = {
    "R2_Score": float(r2),
    "Accuracy": float(accuracy_score(y_test_bin, y_pred_bin)),
    "Precision": float(precision_score(y_test_bin, y_pred_bin)),
    "Recall": float(recall_score(y_test_bin, y_pred_bin)),
    "F1_Score": float(f1_score(y_test_bin, y_pred_bin))
}

for k, v in metrics.items():
    print(f"   {k}: {v:.4f}")

# 4. Save Model
print("\n4. Saving Model for Web App...")
model_path = os.path.join(MODELS_DIR, 'xgb_production.pkl')
joblib.dump(xgb, model_path)
print(f"   Saved to: {model_path}")
print(f"   Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

with open(os.path.join(OUTPUT_DIR, 'production_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n" + "="*50)
print("  TRAINING COMPLETE! MODEL IS READY FOR THE APP.")
print("="*50)

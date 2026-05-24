"""
Step 2: Exploratory Data Analysis & Preprocessing
===================================================
- Load and inspect the dataset
- Distribution analysis for all 20 features
- Correlation heatmap
- Outlier detection (IQR method)
- Feature scaling (StandardScaler)
- Train/test split (80/20)
- Save processed data and scaler
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# Configuration
# ============================================================
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = 'FloodProbability'
FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("  STEP 2: Exploratory Data Analysis & Preprocessing")
print("=" * 60)

print("\n[1] Loading dataset...")
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='id')
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col='id')

print(f"  Train shape: {train.shape}")
print(f"  Test shape:  {test.shape}")
print(f"  Features:    {len(FEATURES)}")
print(f"  Target:      {TARGET}")

# ============================================================
# 2. Basic Statistics
# ============================================================
print("\n[2] Basic statistics...")
print(train.describe().round(3))
print(f"\n  Missing values:\n{train.isnull().sum().to_string()}")
print(f"\n  Data types:\n{train.dtypes.value_counts().to_string()}")

# ============================================================
# 3. Target Distribution
# ============================================================
print("\n[3] Plotting target distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(train[TARGET], bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Flood Probability')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Flood Probability')
axes[0].axvline(train[TARGET].mean(), color='red', linestyle='--', label=f'Mean: {train[TARGET].mean():.3f}')
axes[0].axvline(train[TARGET].median(), color='green', linestyle='--', label=f'Median: {train[TARGET].median():.3f}')
axes[0].legend()

axes[1].boxplot(train[TARGET], vert=True)
axes[1].set_ylabel('Flood Probability')
axes[1].set_title('Box Plot of Flood Probability')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Mean: {train[TARGET].mean():.4f}, Std: {train[TARGET].std():.4f}")
print(f"  Min: {train[TARGET].min():.4f}, Max: {train[TARGET].max():.4f}")

# ============================================================
# 4. Feature Distributions
# ============================================================
print("\n[4] Plotting feature distributions...")
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
for i, feat in enumerate(FEATURES):
    ax = axes[i // 5, i % 5]
    ax.hist(train[feat], bins=30, color=plt.cm.tab20(i/20), edgecolor='white', alpha=0.8)
    ax.set_title(feat, fontsize=10, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. Correlation Heatmap
# ============================================================
print("\n[5] Plotting correlation heatmap...")
corr = train[FEATURES + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 7})
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# Feature-target correlations
print("\n  Feature correlations with FloodProbability:")
target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)
for feat, c in target_corr.items():
    bar = "#" * int(abs(c) * 50)
    print(f"    {feat:35s} {c:+.4f} {bar}")

# ============================================================
# 6. Outlier Detection (IQR)
# ============================================================
print("\n[6] Outlier detection (IQR method)...")
outlier_counts = {}
for feat in FEATURES:
    Q1 = train[feat].quantile(0.25)
    Q3 = train[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((train[feat] < lower) | (train[feat] > upper)).sum()
    outlier_counts[feat] = outliers

print(f"  {'Feature':<35s} {'Outliers':>10s} {'%':>8s}")
print(f"  {'-' * 55}")
for feat, count in sorted(outlier_counts.items(), key=lambda x: -x[1]):
    pct = count / len(train) * 100
    print(f"  {feat:<35s} {count:>10,d} {pct:>7.2f}%")

# Box plots for outliers
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
for i, feat in enumerate(FEATURES):
    ax = axes[i // 5, i % 5]
    ax.boxplot(train[feat], vert=True)
    ax.set_title(feat, fontsize=10, fontweight='bold')
plt.suptitle('Feature Box Plots (Outlier Visualization)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. Feature Scaling & Train/Test Split
# ============================================================
print("\n[7] Splitting and scaling data...")

X = train[FEATURES]
y = train[TARGET]

# Stratified split: bin target into quartiles for stratification
y_binned = pd.qcut(y, q=4, labels=False, duplicates='drop')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y_binned
)

print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape} (mean: {y_train.mean():.4f})")
print(f"  y_test:  {y_test.shape} (mean: {y_test.mean():.4f})")

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)

print(f"\n  Scaler means (first 5): {scaler.mean_[:5].round(3)}")
print(f"  Scaler stds  (first 5): {scaler.scale_[:5].round(3)}")

# ============================================================
# 8. Save Processed Data
# ============================================================
print("\n[8] Saving processed data...")
X_train_scaled.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'))
y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'))

# Also save unscaled for models that don't need scaling (tree-based)
X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train_raw.csv'))
X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test_raw.csv'))

joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
print(f"  Saved to {PROCESSED_DIR}")
print(f"  Scaler saved to {MODELS_DIR}/scaler.pkl")

print("\n" + "=" * 60)
print("  EDA & Preprocessing Complete!")
print("=" * 60)

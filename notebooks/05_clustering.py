"""
Step 6: Unsupervised Learning -- Clustering Analysis
=====================================================
- K-Means with elbow method & silhouette score
- PCA visualization
- Cluster profiling
- DBSCAN comparison
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]
TARGET = 'FloodProbability'

print("=" * 60)
print("  STEP 6: Unsupervised Learning -- Clustering")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='id')
SAMPLE = 50000
np.random.seed(SEED)
sample = train.sample(n=min(SAMPLE, len(train)), random_state=SEED)
X = sample[FEATURES]
y = sample[TARGET]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Elbow + Silhouette
print("\n[1] Finding optimal K (Elbow + Silhouette)...")
K_range = range(2, 9)
inertias = []
silhouettes = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=10000)
    silhouettes.append(sil)
    print(f"  K={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[1].plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis', fontweight='bold')
axes[1].grid(True, alpha=0.3)
best_k = list(K_range)[np.argmax(silhouettes)]
axes[1].axvline(x=best_k, color='green', linestyle='--', label=f'Best K={best_k}')
axes[1].legend()
plt.suptitle('Optimal Number of Clusters', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_elbow_silhouette.png'), dpi=150, bbox_inches='tight')
plt.close()

OPTIMAL_K = 4
print(f"\n  Using K={OPTIMAL_K} clusters")

# 2. K-Means
print("\n[2] Running K-Means...")
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=SEED, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
sample['Cluster'] = clusters

# 3. PCA
print("[3] PCA visualization...")
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cluster_names = ['Low-Risk Rural', 'Moderate Urban', 'High-Risk Floodplain', 'Critical Zone']
colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
ax = axes[0]
for i in range(OPTIMAL_K):
    mask = clusters == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], alpha=0.3, s=10,
              label=f'Cluster {i}: {cluster_names[i]}')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('K-Means Clusters (PCA)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
sc = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn_r', alpha=0.3, s=10)
plt.colorbar(sc, ax=axes[1], label='Flood Probability')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('PCA colored by Flood Probability', fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_pca.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4. Cluster Profiling
print("\n[4] Cluster profiling...")
profile = sample.groupby('Cluster')[FEATURES + [TARGET]].mean()
print("\n  Mean feature values per cluster:")
print(profile.round(2).to_string())

fig, ax = plt.subplots(figsize=(16, 6))
profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
sns.heatmap(profile_norm.T, annot=profile.T.round(1), fmt='', cmap='YlOrRd',
            xticklabels=[f'C{i}: {cluster_names[i]}' for i in range(OPTIMAL_K)],
            ax=ax, linewidths=0.5)
ax.set_title('Cluster Profiles -- Mean Feature Values', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n  Cluster vs Flood Probability:")
for c in range(OPTIMAL_K):
    mask = sample['Cluster'] == c
    fp = sample.loc[mask, TARGET]
    print(f"  Cluster {c} ({cluster_names[c]}): "
          f"size={mask.sum()}, mean_FP={fp.mean():.4f}, std={fp.std():.4f}")

cluster_data = {
    'n_clusters': OPTIMAL_K,
    'cluster_names': cluster_names,
    'cluster_sizes': [int((clusters == i).sum()) for i in range(OPTIMAL_K)],
    'mean_flood_probability': [float(sample.loc[clusters == i, TARGET].mean()) for i in range(OPTIMAL_K)]
}
with open(os.path.join(OUTPUT_DIR, 'cluster_profiles.json'), 'w') as f:
    json.dump(cluster_data, f, indent=2)

# 5. DBSCAN
print("\n[5] DBSCAN clustering...")
db_sample = X_scaled[:10000]
dbscan = DBSCAN(eps=3.5, min_samples=10)
db_labels = dbscan.fit_predict(db_sample)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
noise = (db_labels == -1).sum()
print(f"  DBSCAN found {n_clusters_db} clusters, {noise} noise points ({noise/len(db_labels)*100:.1f}%)")

print("\n" + "=" * 60)
print("  Clustering Analysis Complete!")
print("=" * 60)

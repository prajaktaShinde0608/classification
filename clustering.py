# Clustering Algorithms on a real-world dataset (Wine chemical analysis)
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

# Step 1: Load the dataset (real wines grown in one region of Italy)
wine = load_wine()
X, y_true = wine.data, wine.target
print("Samples:", X.shape[0], "| Features:", X.shape[1])

# Step 2: Scale the features so that no single feature dominates distances
X_scaled = StandardScaler().fit_transform(X)

# Step 3: Choose the number of clusters for K-Means (elbow + silhouette)
print("\nK-MEANS: CHOOSING K")
print("-" * 45)
print(f"{'K':>2} {'Inertia':>10} {'Silhouette':>11} {'Davies-Bouldin':>15}")
ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    s = silhouette_score(X_scaled, km.labels_)
    d = davies_bouldin_score(X_scaled, km.labels_)
    inertias.append(km.inertia_)
    sils.append(s)
    print(f"{k:>2} {km.inertia_:>10.2f} {s:>11.4f} {d:>15.4f}")
best_k = list(ks)[int(np.argmax(sils))]
print("Best K by silhouette score:", best_k)

# Step 4: Fit the four clustering algorithms
K = 3
models = {
    "K-Means": KMeans(n_clusters=K, n_init=10, random_state=42),
    "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=K, linkage="ward"),
    "DBSCAN": DBSCAN(eps=2.3, min_samples=5),
    "Gaussian Mixture": GaussianMixture(n_components=K, random_state=42),
}

labels_all, rows = {}, []
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    labels_all[name] = labels

    mask = labels != -1  # DBSCAN marks noise points as -1
    n_clusters = len(set(labels[mask]))
    n_noise = int((~mask).sum())

    # Internal metrics (computed on clustered points only)
    sil = silhouette_score(X_scaled[mask], labels[mask])
    dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
    chi = calinski_harabasz_score(X_scaled[mask], labels[mask])
    # External check against the known wine cultivars
    ari = adjusted_rand_score(y_true, labels)

    rows.append(
        {
            "Algorithm": name,
            "Clusters": n_clusters,
            "Noise": n_noise,
            "Silhouette": sil,
            "Davies-Bouldin": dbi,
            "Calinski-Harabasz": chi,
            "ARI": ari,
        }
    )

# Step 5: Comparison table
df = pd.DataFrame(rows)
print("\n\n" + "=" * 85)
print("COMPARISON OF CLUSTERING ALGORITHMS")
print("=" * 85)
print(df.round(4).to_string(index=False))

# Step 6: Best algorithm per metric
print("\n\nBEST ALGORITHM ACCORDING TO EACH METRIC")
print("-" * 45)
print("Highest Silhouette        :", df.loc[df["Silhouette"].idxmax(), "Algorithm"])
print("Lowest Davies-Bouldin     :", df.loc[df["Davies-Bouldin"].idxmin(), "Algorithm"])
print("Highest Calinski-Harabasz :", df.loc[df["Calinski-Harabasz"].idxmax(), "Algorithm"])
print("Highest ARI (vs. cultivar):", df.loc[df["ARI"].idxmax(), "Algorithm"])

# Step 7: K-Means cluster profile (mean of key features per cluster)
profile = pd.DataFrame(X, columns=wine.feature_names)
profile["cluster"] = labels_all["K-Means"]
print("\n\nK-MEANS CLUSTER SIZES AND PROFILE")
print("-" * 45)
print(profile.groupby("cluster")[["alcohol", "flavanoids", "color_intensity", "proline"]]
      .agg("mean").round(2).assign(size=profile["cluster"].value_counts().sort_index()).to_string())

# Step 8: Visual inspection with PCA (2 components)
pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X_scaled)
print("\nPCA variance explained by 2 components: %.1f%%" % (100 * pca.explained_variance_ratio_.sum()))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
ax = axes.ravel()
ax[0].plot(list(ks), inertias, "o-")
ax[0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
ax[1].plot(list(ks), sils, "o-", color="tab:green")
ax[1].set(title="Silhouette score vs K", xlabel="K", ylabel="Silhouette")
ax[2].scatter(X2[:, 0], X2[:, 1], c=y_true, cmap="viridis", s=25)
ax[2].set(title="True cultivars (reference)", xlabel="PC1", ylabel="PC2")
for a, name in zip(ax[3:], models):
    a.scatter(X2[:, 0], X2[:, 1], c=labels_all[name], cmap="viridis", s=25)
    a.set(title=name, xlabel="PC1", ylabel="PC2")
plt.tight_layout()
plt.savefig("clusters.png", dpi=130)
print("Saved plot: clusters.png")

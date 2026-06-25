import cluster
import data_loader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def evaluate_clustering(X, labels, centroids, distance_func):
    """
    Evaluating the metrics of the paper: silhoutte, separation, compactness, ratio
    distance_func: either Euclidiean or Wassertein
    """
    k = len(centroids)

    # Silhoutte score
    # for wassertein, sklearn don't provide the metrix so we use the distance matrix
    if distance_func == "wassertein":
        distance_matrix = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                d = wasserstein_distance(X[i], X[j])
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d
        sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
    else:
        sil_score = silhouette_score(X, labels, metric='euclidean')

    # Cluster separation
    separation = 0
    pairs = 0
    for i in range(k):
        for j in range(i+1, k):
            if distance_func == 'wassertein':
                separation += wasserstein_distance(centroids[i], centroids[j])
            else:
                separation += np.linalg.norm(centroids[i] - centroids[j])
            pairs += 1
    separation /= pairs

    # Within cluster compactness
    compactness = 0
    for i in range(k):
        cluster_points = X[labels == i]
        for p in cluster_points:
            if distance_func == 'wassertein':
                compactness += wasserstein_distance(p, centroids[i])
            else:
                compactness += np.linalg.norm(p - centroids[i])
    compactness /= len(X)

    # Separation to compactness ratio
    ratio = separation / compactness if compactness > 0 else 0

    return sil_score, separation, compactness, ratio

subdomains, position = data_loader.domain_decomposition(data_loader.load_data())

# Encode
Z_pca = cluster.pca_reduction(subdomains, 10)
Z_spec = cluster.energy_spectrum_reduction(subdomains, top_p=10)

# Cluster
labels_euclidean_pca, centr_euclidean_pca = cluster.euclidean_kmeans(Z_pca)
labels_wassertein_pca, centr_wassertein_pca = cluster.wassertein_kmeans(Z_pca, k=3)
labels_euclidean_spec, centr_euclidean_spec = cluster.euclidean_kmeans(Z_spec)
labels_wassertein_spec, centr_wassertein_spec = cluster.wassertein_kmeans(Z_spec, k=3)

# Eval
print(f"{'Method': <25}, | {'Silhoutte': <12} |{'Separation': <12}, | {'Compactness': <12}, | {'Ratio': <12}")
print("-"*85)
metrics = evaluate_clustering(Z_pca, labels_euclidean_pca, centr_euclidean_pca, 'euclidean')
print(f"{'PCA + Euclidean': <25} | {metrics[0]} | {metrics[1]} | {metrics[2]} | {metrics[3]}")
metrics = evaluate_clustering(Z_pca, labels_wassertein_pca, centr_wassertein_pca, 'wassertein')
print(f"{'PCA + Wassertein': <25} | {metrics[0]} | {metrics[1]} | {metrics[2]} | {metrics[3]}")
metrics = evaluate_clustering(Z_spec, labels_euclidean_spec, centr_euclidean_spec, 'euclidean')
print(f"{'Spec + Euclidean': <25} | {metrics[0]} | {metrics[1]} | {metrics[2]} | {metrics[3]}")
metrics = evaluate_clustering(Z_spec, labels_wassertein_spec, centr_wassertein_spec, 'wassertein')
print(f"{'Spec + Wassertein': <25} | {metrics[0]} | {metrics[1]} | {metrics[2]} | {metrics[3]}")

# Plots
# Projectio into a 2d space for visualization
Z_pca_2d = PCA(n_components=2).fit_transform(Z_pca)
Z_spec_2d = PCA(n_components=2).fit_transform(Z_spec)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#PCA density plot
sns.kdeplot(x=Z_pca_2d[:,0], y=Z_pca_2d[:,1], cmap='Blues', fill=True, thresh=0.1, ax=axes[0])
axes[0].scatter(Z_pca_2d[:,0], Z_pca_2d[:,1], cmap='viridis', c=labels_euclidean_pca, s=20)
axes[0].set_title("Cluster density: PCA Encoding \n Colored points are the labels")
# Spec density plot
sns.kdeplot(x=Z_spec_2d[:,0], y=Z_spec_2d[:,1], cmap='Blues', fill=True, thresh=0.1, ax=axes[1])
axes[1].scatter(Z_spec_2d[:,0], Z_spec_2d[:,1], cmap='viridis', c=labels_wassertein_spec, s=20)
axes[1].set_title("Cluster density: Energy Spectrum Encoding \n Colored points are the labels")

plt.tight_layout()
plt.show()

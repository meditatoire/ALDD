import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from data_loader import load_data, domain_decomposition
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def pca_reduction(subdomains, n_comp=10):
    X = np.array([sub.flatten() for sub in subdomains]) # Flattening will make us loose info on x and y
    # Note: maybe a 2D pca is better
    pca = PCA(n_components=n_comp)
    Z_pca = pca.fit_transform(X)
    #print(pca.explained_variance_ratio_)
    return Z_pca

def euclidean_kmeans(Z, k=3):
    kmeans_pca = KMeans(n_clusters=k)
    labels_pca = kmeans_pca.fit_predict(Z)
    centroids = kmeans_pca.cluster_centers_
    return labels_pca, centroids

def energy_spectrum_reduction(subdomains, top_p=10):
    features = []
    N = subdomains[0].shape[0] # assuming the shape of the subdomain is (N, N)

    # Compute the radial distance map
    center = N // 2
    y, x = np.indices((N,N))
    r = np.sqrt((x-center)**2 + (y-center)**2)

    # Radial bins (integer distances from 0 to N/2)
    # E.g., for N=32 bins will be 0,1,..,16
    max_k = N//2
    radial_bins = np.arange(0, max_k+1)
    #note: I need to review this since the in the even case we drop the furthest corner
    #also we mix the diagonal energies with axial energies since we "project" the corners
    #into the closest rings

    for sub in subdomains:
        # 2D fourier transform
        f_transform = np.fft.fft2(sub)

        # Shift zero-frequency component to center
        f_shifted = np.fft.fftshift(f_transform)

        # Compute energy
        energy_2d = np.abs(f_shifted)**2

        # For each ring of distance r average the values
        energy_1d = np.zeros(len(radial_bins))
        for i, r_val in enumerate(radial_bins):
            mask = (r >= r_val - 0.5) & (r < r_val + 0.5)
            if np.any(mask):
                energy_1d[i] = np.mean(energy_2d[mask])
        features.append(energy_1d[:top_p])

        # Normalize to one for wassertein distance later
        total_energy = np.sum(energy_1d)
        if total_energy > 0:
            energy_1d = energy_1d / total_energy

    #print(np.array(features).shape, np.array(features)[0])
    return np.array(features)

def wassertein_kmeans(X, k, max_iter=100, tol=1e-4):
    n_samples = X.shape[0]

    # K-means++ initialisation
    centroids = [X[np.random.randint(n_samples)]]
    for _ in range(1, k):
        dists = np.array([
            min([wasserstein_distance(x, c) for c in centroids]) for x in X])
        sq_dists = dists**2
        if np.sum(sq_dists) == 0:
            next_idx = np.random.randint(n_samples)
        else:
            probs = dists**2 / np.sum(dists**2)
            next_idx = np.random.choice(n_samples, p=probs)
        centroids.append(X[next_idx])
    centroids = np.array(centroids)

    # K-means main loop
    labels = np.zeros(n_samples, dtype=int)
    for iter in range(max_iter):
        for i, x in enumerate(X):
            dists = [wasserstein_distance(x, c) for c in centroids]
            labels[i] = np.argmin(dists)

        new_centroids = centroids.copy()
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                min_dists = np.array([
                    min(wasserstein_distance(x, c) for c in centroids) for x in X
                ])
                new_centroids[j] = X[np.argmax(min_dists)]

        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            break
        centroids = new_centroids
    return labels, centroids


# plot of the subdomains labeled as colored rectangles depending on their labels
# for test!

# u = load_data()
# print("u shape:", u.shape)
# block_size = 32
# overlap = 1
# subdomains, coords = domain_decomposition(u, block_size=block_size, overlap=overlap)
# Z = energy_spectrum_reduction(subdomains)
# labels, centroids = wassertein_kmeans(Z, 3)

# fig, ax = plt.subplots(figsize=(7, 7))

# #Field plot
# im = ax.imshow(u, cmap="viridis", origin="upper")
# plt.colorbar(im, ax=ax, label="Field value")

# # Plot cluster rectangles on top
# cluster_cmap = plt.get_cmap("flag", len(np.unique(labels)))
# for (i, j), label in zip(coords, labels):
#     color = cluster_cmap(label)
#     rect = Rectangle(
#         (j - 0.5, i - 0.5),
#         block_size,
#         block_size,
#         fill=False,
#         edgecolor=color,
#         linewidth=2,
#     )
#     ax.add_patch(rect)

# ax.set_xlim(-0.5, u.shape[1] - 0.5)
# ax.set_ylim(u.shape[0] - 0.5, -0.5)
# ax.set_title("Clusters Over a Field")
# ax.set_xlabel("x")
# ax.set_ylabel("y")

# plt.show()

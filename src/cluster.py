import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from data_loader import load_data, domain_decomposition, domain_decomp_single_frame
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SPATIAL_AXES = (0, 1) #La FFT prends par défaut les deux derniers axes comme les axes spatiaux or nous avons (x,y,nb_value) donc nous devons le préciser 

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
    #NOTE: top_p silently fail if max_k < top_p the paper fail to mention top_p values

    features = []
    N = subdomains[0].shape[0] # assuming the shape of the subdomain is (N, N)

    # Carte des distances au centre fréquentiel
    center = N // 2
    y, x = np.indices((N, N))

    radius_map = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    # Rayon maximal atteignant les coins du spectre
    max_radius = int(np.ceil(np.sqrt(2) * N / 2))

    radial_bins = np.arange(0,max_radius + 1)

    number_of_bins = min(top_p,len(radial_bins))

    for sub in subdomains:
        
        #--------------------------------------------------------------#
        # Calcul de E(k_x,k_y) = |\hat{u}|^2 + |\hat{v}|^2 + |\hat{w}|^2
        #--------------------------------------------------------------#
        #on met bien en array 
        field = np.asarray(sub)
        
        #on checke qu'on a bien les bons trucs 
        if field.ndim not in (2, 3):
            raise ValueError(
                "Un sous-domaine doit avoir la shape (N, N) "
                "ou (N, N, C)."
            )

        f_transform = np.fft.fft2(field,axes=SPATIAL_AXES)  # We might do sub-sub.mean() to fix the r0 problem

        # Shift zero-frequency component to center
        f_shifted = np.fft.fftshift(f_transform,axes=SPATIAL_AXES)

        # Compute energy
        energy_2d = np.abs(f_shifted)**2
        
        if field.ndim == 3:
            energy_2d = energy_2d.sum(axis=-1)
            
        #--------------------------------------------------------------#
        #--------------------------------------------------------------#
        
        #--------------------------------------------------------------#
        # somme de l'énergie sur l'anneau fréquentiel 
        #--------------------------------------------------------------#
        # For each ring of distance r sum the values
        energy_1d = np.zeros(len(radial_bins))
        for i, r_val in enumerate(radial_bins):
            mask = (radius_map >= r_val - 0.5) & (radius_map < r_val + 0.5)
            if np.any(mask):
                energy_1d[i] = np.sum(energy_2d[mask])
                   
        #--------------------------------------------------------------#
        # On tronque et on normalise
        #--------------------------------------------------------------#
        spectrum = np.asarray(energy_1d[:number_of_bins],dtype=np.float64)
        total_energy = spectrum.sum()
        features.append(spectrum/total_energy)
        
    return np.stack(features) 


def spectrum_wasserstein(x, y):
    # 1D W_2^2 distance between two normalized spectra (PDFs on integer bins).
    Qx = _quantile_function(x)
    Qy = _quantile_function(y)
    return _wasserstein2_q(Qx, Qy)


# Shared quantile grid for 1D Wasserstein computations.
_N_QUANTILES = 2000
_T_GRID = (np.arange(_N_QUANTILES) + 0.5) / _N_QUANTILES


def _quantile_function(pdf):
    # Quantile function Q(t) of a normalized PDF on integer bins 0..n-1.
    # Returns values in [0, n_bins-1] evaluated on the shared t-grid.
    cdf = np.cumsum(pdf)
    return np.searchsorted(cdf, _T_GRID, side="left").astype(float)


def _wasserstein2_q(Qx, Qy):
    # W_2^2 distance between two quantile functions on the shared t-grid.
    return np.mean((Qx - Qy) ** 2)


def _quantile_to_pdf(Q_bar, n_bins):
    # Project a barycenter quantile function back onto integer bins (for plotting).
    hist, _ = np.histogram(Q_bar, bins=n_bins, range=(-0.5, n_bins - 0.5))
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist

def wasserstein2_distance_matrix(X):
    # Pairwise W_2^2 distances between normalized spectra in X (n_samples, n_bins).
    # Quantile functions are computed once and the distance is vectorized per row.
    Q = np.array([_quantile_function(x) for x in X])
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = ((Q[i] - Q) ** 2).mean(axis=1)
    return D


def wassertein_barycenter(points):
    # Exact 1D W_2 barycenter of normalized spectra, returned as a PDF.
    # Implemented as the MEAN OF QUANTILE FUNCTIONS (the W_2 Frechet mean),
    # NOT the arithmetic mean of the PDFs.
    Qs = np.array([_quantile_function(p) for p in points])
    Q_bar = Qs.mean(axis=0)
    return _quantile_to_pdf(Q_bar, points.shape[1])


def wassertein_kmeans(X, k, max_iter=100, tol=1e-4):
    n_samples = X.shape[0]
    Q = np.array([_quantile_function(x) for x in X])  # (n_samples, n_quantiles)

    # K-means++ initialisation using the W_2 distance.
    centroids_Q = [Q[np.random.randint(n_samples)]]
    for _ in range(1, k):
        dists = np.array([
            min(_wasserstein2_q(q, c) for c in centroids_Q) for q in Q])
        if np.sum(dists ** 2) == 0:
            next_idx = np.random.randint(n_samples)
        else:
            probs = dists / np.sum(dists)
            next_idx = np.random.choice(n_samples, p=probs)
        centroids_Q.append(Q[next_idx])

    # K-means main loop
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        for i in range(n_samples):
            dists = [_wasserstein2_q(Q[i], c) for c in centroids_Q]
            labels[i] = np.argmin(dists)

        new_centroids_Q = list(centroids_Q)
        for j in range(k):
            cluster_points = Q[labels == j]
            if len(cluster_points) > 0:
                new_centroids_Q[j] = cluster_points.mean(axis=0)
            else:
                min_dists = np.array([
                    min(_wasserstein2_q(q, c) for c in centroids_Q) for q in Q])
                new_centroids_Q[j] = Q[np.argmax(min_dists)]

        shift = sum(_wasserstein2_q(new_centroids_Q[j], centroids_Q[j]) for j in range(k))
        if shift < tol:
            break
        centroids_Q = new_centroids_Q

    # Return centroids as PDFs (projected back) for downstream use / plotting.
    centroids = np.array([_quantile_to_pdf(c, X.shape[1]) for c in centroids_Q])
    return labels, centroids


# #plot of the subdomains labeled as colored rectangles depending on their labels
# #for test!

# time_series = load_data(num_timesteps=1)
# u = time_series[0]
# print("u shape:", u.shape)
# block_size = 32
# overlap = 1
# subdomains = domain_decomp_single_frame(u, block_size=block_size, overlap=overlap)
# step = block_size - overlap
# coords = [
#     (i, j)
#     for i in range(0, u.shape[0] - block_size + 1, step)
#     for j in range(0, u.shape[1] - block_size + 1, step)
# ]
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

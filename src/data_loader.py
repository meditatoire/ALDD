import os
import urllib.request
import scipy.io
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """
    We load the Karman Vortex Dataset (flow past a cylinder)
    and interpolate it to a regular grid.
    """
    filename="./data/cylinder_nektar_wake.mat"
    url = "https://github.com/maziarraissi/PINNs/raw/master/main/Data/cylinder_nektar_wake.mat"

    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        print(f"Data downloaded in {filename}")

    data = scipy.io.loadmat(filename)

    # U contains [u,v] velocity components
    X = data['X_star'] #shape: (N, 2)
    U = data['U_star'] #shape: (N, 2, T)

    x_coords = X[:, 0]
    y_coords = X[:, 1]
    u_velocity = U[:, 0, 150]

    # we create a uniform grid of 256x128 points
    grid_x, grid_y = np.mgrid[min(x_coords):max(x_coords):256j, min(y_coords):max(y_coords):128j]

    # we interpolate the scattered data to the grid
    grid_u = griddata((x_coords, y_coords), u_velocity, (grid_x, grid_y), method='cubic')

    #Filling NaN values with 0 (In case the grid extends outside the boundary of the scattered points)
    grid_u = np.nan_to_num(grid_u, nan=0.0)

    # Plot
    #plt.pcolormesh(grid_x, grid_y, grid_u, shading='auto', cmap='viridis')
    #plt.scatter(x_coords, y_coords, color='red', edgecolors='black', s=20, label='Original Data')
    #plt.legend()
    #plt.title("Interpolated Grid vs Original Scattered Points")
    #plt.show()

    return grid_u.T # Transpose to get shape (128, 256)

def domain_decomposition(u, block_size=32, overlap=1):
    H, W = u.shape
    subdomains = []
    positions = []
    step = block_size - overlap
    for i in range(0, H-block_size+1, step):
        for j in range(0, W-block_size+1, step):
            subdomain = u[i:i+block_size, j:j+block_size]
            subdomains.append(subdomain)
            positions.append((i,j))
    return subdomains, positions

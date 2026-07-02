import data_loader
import cluster
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from models import DeepONet_BENO, FNO_BENO


# Configuration and hyperparameters
MODEL_TYPE = 'fno'  # model: 'fno' or 'deeponet'

N = 16     # Subdomain size (NxN)
BATCH_SIZE = 64
EPOCHS = 30
K_CLUSTER = 3
TOP_P = 25 # Dimension reduction for subdomain clustering
VAL_TIMESTEPS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def subdomain_union(subdomains, H, W, block_size=N, overlap=1):
    """Reconstructs the full domain from subdomains, averaging the overlaps."""
    step = block_size - overlap
    full_domain = np.zeros((H, W))
    count = np.zeros((H, W))

    idx = 0
    for i in range(0, H - block_size + 1, step):
        for j in range(0, W - block_size + 1, step):
            full_domain[i:i+block_size, j:j+block_size] += subdomains[idx]
            count[i:i+block_size, j:j+block_size] += 1
            idx += 1
    return full_domain / np.maximum(count, 1)

def setup_grids():
    """Generates the appropriate grid formats for FNO and DeepONet."""
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    if MODEL_TYPE == 'fno':
        # FNO expects grid as additional channels: (Batch, 2, H, W)
        fno_grid_np = np.stack([X, Y], axis=0) # Shape: (2, N, N)
        return torch.tensor(fno_grid_np, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        # DeepONet expects flattened coordinates: (Num_points, 2)
        grid_coords = np.stack([X.flatten(), Y.flatten()], axis=1)
        return torch.tensor(grid_coords, dtype=torch.float32).to(device)
# NOTE: in the future might be better to move sub_union and setup_grids to data_loader.py or some setup.py

def initialize_model():
    """Initializes the selected model."""
    if MODEL_TYPE == 'fno':
        return FNO_BENO(boundary_size=N*4-4, modes=8, width=64, num_layers=4).to(device)
    else:
        return DeepONet_BENO(branch_input_dim=N*N, latent_dim=128).to(device)


# Training loop
def train_models(u_t_all, u_t1_all, labels, grid_data):
    trained_models = []

    for cluster_idx in range(K_CLUSTER):
        cluster_u_t = u_t_all[labels == cluster_idx]
        cluster_u_t1 = u_t1_all[labels == cluster_idx]
        cluster_boundaries = np.array([data_loader.extract_boundary(sub) for sub in cluster_u_t])

        print(f"Training {MODEL_TYPE.upper()} for Cluster {cluster_idx} ({len(cluster_u_t)} samples)")

        # Format Tensors based on model type
        boundary_tensor = torch.tensor(cluster_boundaries, dtype=torch.float32).unsqueeze(-1).to(device)

        if MODEL_TYPE == 'fno':
            u_t_tensor = torch.tensor(cluster_u_t, dtype=torch.float32).unsqueeze(1).to(device)
            u_t1_tensor = torch.tensor(cluster_u_t1, dtype=torch.float32).unsqueeze(1).to(device)
        else:
            u_t_tensor = torch.tensor([sub.flatten() for sub in cluster_u_t], dtype=torch.float32).to(device)
            u_t1_tensor = torch.tensor([sub.flatten() for sub in cluster_u_t1], dtype=torch.float32).to(device)

        model = initialize_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_func = nn.MSELoss()

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            for i in range(0, len(u_t_tensor), BATCH_SIZE):
                batch_u_t = u_t_tensor[i:i+BATCH_SIZE]
                batch_u_t1 = u_t1_tensor[i:i+BATCH_SIZE]
                batch_boundaries = boundary_tensor[i:i+BATCH_SIZE]

                optimizer.zero_grad()

                if MODEL_TYPE == 'fno':
                    batch_grid = grid_data.expand(len(batch_u_t), -1, -1, -1)
                    predictions = model.forward(batch_u_t, batch_grid, batch_boundaries)
                else:
                    predictions = model.forward(batch_u_t, grid_data, batch_boundaries)

                loss = loss_func(predictions, batch_u_t1)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / (len(u_t_tensor) / BATCH_SIZE):.6f}")

        trained_models.append(model)

    return trained_models

# Validation
def validate_autoregressive(trained_models, centroids, grid_data):
    print(f"Validation up to T+{VAL_TIMESTEPS}")
    val_series = data_loader.load_data(num_timesteps=VAL_TIMESTEPS, start_t=120)
    H, W = val_series.shape[1], val_series.shape[2]

    current_frame = val_series[0]
    prediction_series = []

    for t in range(VAL_TIMESTEPS - 1):
        sub_t = data_loader.domain_decomp_single_frame(current_frame, N)
        Z_val = cluster.energy_spectrum_reduction(sub_t, top_p=TOP_P)
        val_labels = np.array([np.argmin([wasserstein_distance(z, c) for c in centroids]) for z in Z_val])

        predicted_subs = np.zeros_like(sub_t)
        for i, sub in enumerate(sub_t):
            model = trained_models[val_labels[i]]
            model.eval()

            bound = data_loader.extract_boundary(sub)
            boundary_tensor = torch.tensor(bound, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

            with torch.no_grad():
                if MODEL_TYPE == 'fno':
                    sub_tensor = torch.tensor(sub, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    pred = model(sub_tensor, grid_data, boundary_tensor)
                    predicted_subs[i] = pred.squeeze().cpu().numpy()
                else:
                    sub_tensor = torch.tensor(sub.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                    pred = model(sub_tensor, grid_data, boundary_tensor)
                    predicted_subs[i] = pred.squeeze().cpu().numpy().reshape(N, N)

        next_frame = subdomain_union(predicted_subs, H, W, block_size=N, overlap=1)
        prediction_series.append(next_frame)
        current_frame = next_frame

        true_frame = val_series[t+1]
        ss_res = np.sum((true_frame - next_frame)**2)
        ss_total = np.sum((true_frame - np.mean(true_frame))**2)
        r2 = 1 - ss_res / ss_total
        print(f"Step: {t+1}/{VAL_TIMESTEPS-1} | R² Score: {r2:.4f}")

    # Plotting results
    true_final = val_series[-1]
    pred_final = prediction_series[-1]
    error_map = np.abs(true_final - pred_final)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(true_final, cmap='viridis')
    axes[0].set_title(f"Ground Truth (T+{VAL_TIMESTEPS-1})")
    axes[1].imshow(pred_final, cmap='viridis')
    axes[1].set_title(f"{MODEL_TYPE.upper()} Prediction (T+{VAL_TIMESTEPS-1})")
    im3 = axes[2].imshow(error_map, cmap='magma')
    axes[2].set_title("Absolute Error")
    fig.colorbar(im3, ax=axes[2])
    plt.show()


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Model Type: {MODEL_TYPE.upper()}")

    # Setup grids
    grid_data = setup_grids()

    # Load data and cluster
    time_series = data_loader.load_data(num_timesteps=100, start_t=0)
    u_t_all, u_t1_all = data_loader.domain_decomposition(time_series, N)

    Z_spec = cluster.energy_spectrum_reduction(u_t_all, top_p=TOP_P)
    labels, centroids = cluster.wassertein_kmeans(Z_spec, K_CLUSTER)

    # Training
    trained_models = train_models(u_t_all, u_t1_all, labels, grid_data)

    # Validation
    validate_autoregressive(trained_models, centroids, grid_data)

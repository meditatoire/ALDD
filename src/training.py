from numpy.random.mtrand import permutation

import data_loader
import cluster
import torch
import torch.nn as nn
import numpy as np
from pipeline import initialize_model, save_checkpoint, setup_grids


# Configuration and hyperparameters
MODEL_TYPE = 'fno'  # model: 'fno' or 'deeponet'

DATASET = "jhtdb"  # "cylinder" or "jhtdb"
JHTDB_PATH = "./JHTDB/data/jhtdb_test/small_planes.h5"
TRAIN_Z = [64, 512]  # Hold out the remaining z planes for validation.
TRAIN_START = 0
TRAIN_STOP = None

N = 16     # Subdomain size (NxN)
BATCH_SIZE = 512
EPOCHS = 100
K_CLUSTER = 3
TOP_P = 25 # Dimension reduction for subdomain clustering

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training loop
def train_models(u_t_all, u_t1_all, labels, grid_data):
    trained_models = []

    for cluster_idx in range(K_CLUSTER):
        cluster_u_t = u_t_all[labels == cluster_idx]
        cluster_u_t1 = u_t1_all[labels == cluster_idx]
        cluster_boundary_vals = []
        cluster_boundary_xy = []
        for sub in cluster_u_t:
            vals, xy = data_loader.extract_boundary(sub)
            cluster_boundary_vals.append(vals)
            cluster_boundary_xy.append(xy)

        print(f"Training {MODEL_TYPE.upper()} for Cluster {cluster_idx} ({len(cluster_u_t)} samples)")

        # Format Tensors based on model type
        vals_tensor = torch.tensor(np.stack(cluster_boundary_vals), dtype=torch.float32)
        xy_tensor   = torch.tensor(np.stack(cluster_boundary_xy), dtype=torch.float32)
        boundary_tensor = torch.cat([xy_tensor, vals_tensor.unsqueeze(-1)], dim=-1).to(device)

        if MODEL_TYPE == 'fno':
            u_t_tensor = torch.tensor(cluster_u_t, dtype=torch.float32).unsqueeze(1).to(device)
            u_t1_tensor = torch.tensor(cluster_u_t1, dtype=torch.float32).unsqueeze(1).to(device)
        else:
            u_t_tensor = torch.tensor([sub.flatten() for sub in cluster_u_t], dtype=torch.float32).to(device)
            u_t1_tensor = torch.tensor([sub.flatten() for sub in cluster_u_t1], dtype=torch.float32).to(device)

        model = initialize_model(MODEL_TYPE, N, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_func = nn.MSELoss()

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            permutation = torch.randperm(len(u_t_tensor))
            for i in range(0, len(u_t_tensor), BATCH_SIZE):
                idx = permutation[i:i+BATCH_SIZE]
                batch_u_t = u_t_tensor[idx]
                batch_u_t1 = u_t1_tensor[idx]
                batch_boundaries = boundary_tensor[idx]

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

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / (len(u_t_tensor) / BATCH_SIZE):.6f}")

        trained_models.append(model)

    return trained_models


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Model Type: {MODEL_TYPE.upper()}")

    # Setup grids
    grid_data = setup_grids(MODEL_TYPE, N, device)

    # Load data and cluster
    if DATASET == "jhtdb":
        trajectories, z_indices, _ = data_loader.load_jhtdb_data(
            JHTDB_PATH,
            z_indices=TRAIN_Z,
            start_t=TRAIN_START,
            stop_t=TRAIN_STOP,
        )
        print(f"Training z planes: {z_indices}")
        u_t_all, u_t1_all = data_loader.domain_decomposition_trajectories(
            trajectories, N
        )
    else:
        time_series = data_loader.load_data(num_timesteps=100, start_t=0)
        u_t_all, u_t1_all = data_loader.domain_decomposition(time_series, N)

    Z_spec = cluster.energy_spectrum_reduction(u_t_all, top_p=TOP_P)
    labels, centroids = cluster.wassertein_kmeans(Z_spec, K_CLUSTER)

    # Training
    trained_models = train_models(u_t_all, u_t1_all, labels, grid_data)

    checkpoint_path = save_checkpoint(
        trained_models=trained_models,
        centroids=centroids,
        model_type=MODEL_TYPE,
        n=N,
        epochs=EPOCHS,
        k_cluster=K_CLUSTER,
        top_p=TOP_P,
    )
    print(f"Saved checkpoint: {checkpoint_path}")

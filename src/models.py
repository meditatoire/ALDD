import data_loader
import cluster
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance



class DeepONet(nn.Module):
    #Appendix A.2 for hyperparameters
    def __init__(self, branch_input_dim, latent_dim, trunk_input_dim=2, hidden_dim=64, output_dim=1):
        """
        branch_input_dim: Number of observations in a subdomain e.g. 32x32 flattened
        trunk_input_dim: 2 for x and y
        latent_dim: dim of the embedings before the dot product ??
        output_dim: Num of physical variables to predict (here one the velocity u)
        """
        super().__init__()

        # Branch network
        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Trunk network
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Final layer
        self.fc_out = nn.Linear(latent_dim, output_dim)

        # Glorot Normalization Appendix A.2
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u_branch, y_trunk):
        """
        u_branch: (batch_size, branch_input_dim)
        y_trunk: (num_points, trunk_input_dim)
        """
        branch_out = self.branch(u_branch)
        trunk_out = self.trunk(y_trunk)

        dot_product = torch.matmul(branch_out, trunk_out.T)

        return dot_product

N = 16
BATCH_SIZE = 16 #num of subdomains in a batch
EPOCHS = 30
K_CLUSTER = 3
TOP_P = 25 # top-p energies, for dimension reduction

# Trunk input 32x32 grid normalized for better training
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
# flaten to shape(1024, 2)
grid_coords = np.stack([X.flatten(), Y.flatten()], axis=1)
grid_coords_tensor = torch.tensor(grid_coords, dtype=torch.float32)

time_series = data_loader.load_data(num_timesteps=100, start_t=0)
u_t_all, u_t1_all = data_loader.domain_decomposition(time_series, N)

#Encode and cluster
Z_spec = cluster.energy_spectrum_reduction(u_t_all, top_p=TOP_P)
labels, centroids = cluster.wassertein_kmeans(Z_spec, K_CLUSTER)
trained_models = []

for cluster_idx in range(K_CLUSTER):
    cluster_u_t = u_t_all[labels==cluster_idx]
    cluster_u_t1 = u_t1_all[labels==cluster_idx]

    u_t_tensor = torch.tensor([sub.flatten() for sub in cluster_u_t], dtype=torch.float32)
    u_t1_tensor = torch.tensor([sub.flatten() for sub in cluster_u_t1], dtype=torch.float32)

    model = DeepONet(branch_input_dim=N*N, trunk_input_dim=2, latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_func = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for i in range(0, len(u_t_tensor), BATCH_SIZE):
            batch_u_t = u_t_tensor[i:i+BATCH_SIZE]
            batch_u_t1 = u_t1_tensor[i:i+BATCH_SIZE]

            optimizer.zero_grad()
            predictions = model.forward(batch_u_t, grid_coords_tensor)
            loss = loss_func(predictions, batch_u_t1)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / (len(u_t_tensor) /BATCH_SIZE)}")
    trained_models.append(model)

def subdomain_union(subdomains, H, W, block_size=N, overlap=1):
    """Domain reconstruction for predictions (overlaping zones are averaged)"""
    step = block_size - overlap
    full_domain = np.zeros((H,W))
    count = np.zeros((H,W))

    idx=0
    for i in range(0, H-block_size+1, step):
        for j in range(0, W-block_size+1, step):
            full_domain[i:i+block_size, j:j+block_size] += subdomains[idx]
            count[i:i+block_size, j:j+block_size] += 1   #2 in overlapping boundaries 1 elsewhere
            idx += 1
    return full_domain/np.maximum(count, 1)

print("Validation")
VAL_TIMESTEPS = 10
val_series = data_loader.load_data(num_timesteps=VAL_TIMESTEPS, start_t=120)
H, W = val_series.shape[1], val_series.shape[2]

current_frame = val_series[0]
prediction_series = []
for t in range(VAL_TIMESTEPS-1):
    sub_t = data_loader.domain_decomp_single_frame(current_frame, N)
    Z_val = cluster.energy_spectrum_reduction(sub_t, top_p=TOP_P)
    val_labels = np.array([np.argmin([wasserstein_distance(z, c) for c in centroids]) for z in Z_val])

    #subdomain prediction
    predicted_subs = np.zeros_like(sub_t)
    for i, sub in enumerate(sub_t):
        model = trained_models[val_labels[i]]
        model.eval()
        sub_tensor = torch.tensor(sub.flatten(), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(sub_tensor, grid_coords_tensor)

        predicted_subs[i] = pred.squeeze().numpy().reshape(N,N)

    next_frame = subdomain_union(predicted_subs, H, W, block_size=N, overlap=1)
    prediction_series.append(next_frame)
    current_frame = next_frame

    # calcultate the R² score for this timestep
    true_frame = val_series[t+1]
    ss_res = np.sum((true_frame - next_frame)**2)
    ss_total = np.sum((true_frame - np.mean(true_frame))**2)
    r2 = 1 - ss_res / ss_total
    print(f"Step: {t+1}/{VAL_TIMESTEPS} | R² Score: {r2}")

true_final = val_series[-1]
pred_final = prediction_series[-1]
error_map = np.abs(true_final - pred_final)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(true_final, cmap='viridis')
axes[0].set_title("Ground truth T+10")
axes[1].imshow(pred_final, cmap='viridis')
axes[1].set_title("Prediction T+10")
im3 = axes[2].imshow(error_map, cmap='magma')
axes[2].set_title("Absolute error")
fig.colorbar(im3, ax=axes[2])
plt.show()

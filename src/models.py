from scipy.integrate._ivp.radau import E

import data_loader
import cluster
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

N = 32
BATCH_SIZE = 16 #num of subdomains in a batch
EPOCHS = 10
K_CLUSTER = 3

# Trunk input 32x32 grid normalized for better training
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
# flaten to shape(1024, 2)
grid_coords = np.stack([X.flatten(), Y.flatten()], axis=1)
grid_coords_tensor = torch.tensor(grid_coords, dtype=torch.float32)

time_series = data_loader.load_data(num_timesteps=100, start_t=0)
u_t_all, u_t1_all = data_loader.domain_decomposition(time_series)

#Encode and cluster
Z_spec = cluster.energy_spectrum_reduction(u_t_all)
labels, _ = cluster.wassertein_kmeans(Z_spec, K_CLUSTER)

for cluster in range(K_CLUSTER):
    cluster_u_t = u_t_all[labels==cluster]
    cluster_u_t1 = u_t1_all[labels==cluster]

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

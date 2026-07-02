import data_loader
import cluster
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Save tensor as model state

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:x.size(1), :].unsqueeze(0)



class DeepONet_BENO(nn.Module):
    #Appendix A.2 for hyperparameters
    def __init__(self, branch_input_dim, latent_dim, trunk_input_dim=2, hidden_dim=64, output_dim=1, n_heads=2):
        """
        branch_input_dim: Number of observations in a subdomain e.g. 32x32 flattened
        trunk_input_dim: 2 for x and y
        latent_dim: dim of the embedings before the dot product ??
        output_dim: Num of physical variables to predict (here one the velocity u)
        """
        super().__init__()

        #Boundary transformer BENO
        self.beno_conv = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding='same')
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Project transformer output to latent dim
        self.beno_proj = nn.Linear(hidden_dim, latent_dim)

        # Branch network (Now it takes interior + boundary embeddings)
        # We concatenate branch output and embeddings
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

    def forward(self, u_branch, y_trunk, boundary_u):
        """
        u_branch: (batch_size, branch_input_dim)
        y_trunk: (num_points, trunk_input_dim)
        boundary_u: (batch_size, block_size*4 -4, 1)
        """

        # Conv1D expect (batch_size, channels, length)
        beno_conv_out = self.beno_conv(boundary_u.transpose(1,2)).transpose(1,2) # (batch, boudary_size, hidden_dim)

        beno_embed = self.transformer_encoder(self.pos_encoder(beno_conv_out)) # (batch, boudary_size, hidden_dim)

        #average over the seq_len to get a fixed size vector
        beno_embed = beno_embed.mean(dim=1) # (batch, hidden_dim)
        beno_embed = self.beno_proj(beno_embed)

        # branch and trunk
        branch_out = self.branch(u_branch)
        trunk_out = self.trunk(y_trunk)

        #combined branch
        combined_branch = branch_out + beno_embed

        dot_product = torch.matmul(combined_branch, trunk_out.T)

        return dot_product

N = 16
BATCH_SIZE = 32 #num of subdomains in a batch
EPOCHS = 30
K_CLUSTER = 3
TOP_P = 15 # top-p energies, for dimension reduction

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Trunk input 32x32 grid normalized for better training
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
# flaten to shape(1024, 2)
grid_coords = np.stack([X.flatten(), Y.flatten()], axis=1)
grid_coords_tensor = torch.tensor(grid_coords, dtype=torch.float32).to(device)

time_series = data_loader.load_data(num_timesteps=100, start_t=0)
u_t_all, u_t1_all = data_loader.domain_decomposition(time_series, N)

#Encode and cluster
Z_spec = cluster.energy_spectrum_reduction(u_t_all, top_p=TOP_P)
labels, centroids = cluster.wassertein_kmeans(Z_spec, K_CLUSTER)
trained_models = []

for cluster_idx in range(K_CLUSTER):
    cluster_u_t = u_t_all[labels==cluster_idx]
    cluster_u_t1 = u_t1_all[labels==cluster_idx]
    cluster_boundaries = np.array([data_loader.extract_boundary(sub) for sub in cluster_u_t])

    u_t_tensor = torch.tensor(np.array([sub.flatten() for sub in cluster_u_t]), dtype=torch.float32).to(device)
    u_t1_tensor = torch.tensor(np.array([sub.flatten() for sub in cluster_u_t1]), dtype=torch.float32).to(device)
    boundary_tensor = torch.tensor(cluster_boundaries, dtype=torch.float32).unsqueeze(-1).to(device)

    model = DeepONet_BENO(branch_input_dim=N*N, trunk_input_dim=2, latent_dim=128).to(device)
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
            predictions = model.forward(batch_u_t, grid_coords_tensor, batch_boundaries)
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
        sub_tensor = torch.tensor(sub.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        bound = data_loader.extract_boundary(sub)
        boundary_tensor = torch.tensor(bound, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            pred = model(sub_tensor, grid_coords_tensor, boundary_tensor)

        predicted_subs[i] = pred.squeeze().cpu().numpy().reshape(N,N)

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

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import cluster
import data_loader
from models import DeepONet_BENO, FNO_BENO


def subdomain_union(subdomains, H, W, block_size, overlap=1):
    """
    subdomains shape:
        (num_blocks, block_size, block_size)       scalar
        (num_blocks, block_size, block_size, C)    vector
    """
    if subdomains.ndim not in (3, 4):
        raise ValueError("subdomains must have shape (S, N, N) or (S, N, N, C)")

    field_shape = (H, W) + tuple(subdomains.shape[3:])
    full_domain = np.zeros(field_shape, dtype=subdomains.dtype)
    count = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for i in data_loader.block_starts(H, block_size, overlap):
        for j in data_loader.block_starts(W, block_size, overlap):
            full_domain[i:i+block_size, j:j+block_size, ...] += subdomains[idx]
            count[i:i+block_size, j:j+block_size] += 1
            idx += 1

    divisor = np.maximum(count, 1)
    if full_domain.ndim > 2:
        divisor = divisor[..., None]
    return full_domain / divisor


def setup_grids(model_type, n, device):
    """Generates the appropriate grid formats for FNO and DeepONet."""
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    if model_type == 'fno':
        fno_grid_np = np.stack([X, Y], axis=0)
        return torch.tensor(fno_grid_np, dtype=torch.float32).unsqueeze(0).to(device)

    grid_coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    return torch.tensor(grid_coords, dtype=torch.float32).to(device)


def initialize_model(model_type, n, device, channels=1):
    """Initializes the selected model."""
    if model_type == 'fno':
        return FNO_BENO(modes=8, width=64, num_layers=4, field_channels=channels, out_channels=channels).to(device)

    return DeepONet_BENO(branch_input_dim=n*n*channels, latent_dim=128, channels=channels).to(device)


def build_checkpoint_name(model_type, epochs, k_cluster, n, top_p):
    return f"{model_type}_e{epochs}_k{k_cluster}_n{n}_top{top_p}.pt"


def save_checkpoint(
    trained_models,
    centroids,
    model_type,
    n,
    epochs,
    k_cluster,
    top_p,
    output_dir="models",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_type": model_type,
        "n": n,
        "epochs": epochs,
        "k_cluster": k_cluster,
        "top_p": top_p,
        "channels": getattr(trained_models[0], "channels", 1),
        "centroids": centroids,
        "model_state_dicts": [model.state_dict() for model in trained_models],
    }

    checkpoint_path = output_path / build_checkpoint_name(
        model_type=model_type,
        epochs=epochs,
        k_cluster=k_cluster,
        n=n,
        top_p=top_p,
    )
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    models = []

    for state_dict in checkpoint["model_state_dicts"]:
        model = initialize_model(
            checkpoint["model_type"], checkpoint["n"], device,
            channels=checkpoint.get("channels", 1),
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return checkpoint, models


def validate_autoregressive(
    trained_models,
    centroids,
    grid_data,
    model_type,
    n,
    top_p,
    val_timesteps,
    device,
    dataset="cylinder",
    data_path=None,
    val_z=None,
):
    print(f"Validation up to T+{val_timesteps}")
    if dataset == "jhtdb":
        val_series, z_indices, _ = data_loader.load_jhtdb_data(
            data_path,
            z_indices=val_z,
            stop_t=val_timesteps,
            component=("u", "v", "w"),
        )
    else:
        val_series = data_loader.load_data(num_timesteps=val_timesteps, start_t=120)
        z_indices = [None]

    if dataset == "cylinder":
        val_series = val_series[None, ...]

    for z_position, series in enumerate(val_series):
        if dataset == "jhtdb":
            print(f"Validating z={z_indices[z_position]}")
        _validate_series(
            series,
            trained_models,
            centroids,
            grid_data,
            model_type,
            n,
            top_p,
            val_timesteps,
            device,
        )


def _validate_series(
    val_series,
    trained_models,
    centroids,
    grid_data,
    model_type,
    n,
    top_p,
    val_timesteps,
    device,
):
    H, W = val_series.shape[1], val_series.shape[2]

    current_frame = val_series[0]
    prediction_series = []

    for t in range(val_timesteps - 1):
        sub_t = data_loader.domain_decomp_single_frame(current_frame, n)
        Z_val = cluster.energy_spectrum_reduction(sub_t, top_p=top_p)
        val_labels = np.array([
            np.argmin([cluster.spectrum_wasserstein(z, c) for c in centroids])
            for z in Z_val
        ])

        predicted_subs = np.zeros_like(sub_t)
        for i, sub in enumerate(sub_t):
            model = trained_models[val_labels[i]]

            vals, xy = data_loader.extract_boundary(sub)
            if vals.ndim == 1:
                vals = vals[:, None]
            boundary_token = np.concatenate([xy, vals], axis=1)
            boundary_tensor = torch.tensor(boundary_token, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                if model_type == 'fno':
                    sub_tensor = torch.tensor(sub, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                    pred = model(sub_tensor, grid_data, boundary_tensor)
                    predicted_subs[i] = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
                else:
                    sub_tensor = torch.tensor(sub.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                    pred = model(sub_tensor, grid_data, boundary_tensor)
                    predicted_subs[i] = pred.reshape(1, -1, n * n).permute(0, 2, 1).reshape(n, n, -1).squeeze(0).cpu().numpy()

        next_frame = subdomain_union(predicted_subs, H, W, block_size=n, overlap=1)
        prediction_series.append(next_frame)
        current_frame = next_frame

        true_frame = val_series[t+1]
        true_u = true_frame[..., 0] if true_frame.ndim == 3 else true_frame
        next_u = next_frame[..., 0] if next_frame.ndim == 3 else next_frame
        ss_res = np.sum((true_u - next_u)**2)
        ss_total = np.sum((true_u - np.mean(true_u))**2)
        r2 = 1 - ss_res / ss_total
        print(f"Step: {t+1}/{val_timesteps-1} | R² Score: {r2:.4f}")

    true_final = val_series[-1]
    pred_final = prediction_series[-1]
    true_u = true_final[..., 0] if true_final.ndim == 3 else true_final
    pred_u = pred_final[..., 0] if pred_final.ndim == 3 else pred_final
    error_map = np.abs(true_u - pred_u)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(true_u, cmap='viridis')
    axes[0].set_title(f"Ground Truth (T+{val_timesteps-1})")
    axes[1].imshow(pred_u, cmap='viridis')
    axes[1].set_title(f"{model_type.upper()} Prediction (T+{val_timesteps-1})")
    im3 = axes[2].imshow(error_map, cmap='magma')
    axes[2].set_title("Absolute Error")
    fig.colorbar(im3, ax=axes[2])
    plt.show()

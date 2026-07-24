import argparse

import torch

from pipeline import load_checkpoint, setup_grids, validate_autoregressive


VAL_TIMESTEPS = 10
DATASET = "jhtdb"  # "cylinder" or "jhtdb"
JHTDB_PATH = "../JHTDB/data/jhtdb_test/small_planes.h5"
VAL_Z = [288]  # Different z planes than training.py: TRAIN_Z.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a trained ALDD checkpoint.")
    parser.add_argument(
        "checkpoint",
        help="Path to a checkpoint saved by training.py, for example models/fno_e30_k3_n16_top25.pt",
    )
    parser.add_argument(
        "--val-timesteps",
        type=int,
        default=VAL_TIMESTEPS,
        help=f"Number of validation timesteps. Default: {VAL_TIMESTEPS}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint, trained_models = load_checkpoint(args.checkpoint, device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model Type: {checkpoint['model_type'].upper()}")
    print(f"Epochs: {checkpoint['epochs']}")
    print(f"Number of clusters: {checkpoint['k_cluster']}")
    print(f"Subdomain size: {checkpoint['n']}")
    print(f"Spectral bins: {checkpoint['top_p']}")

    grid_data = setup_grids(checkpoint["model_type"], checkpoint["n"], device)
    validate_autoregressive(
        trained_models=trained_models,
        centroids=checkpoint["centroids"],
        grid_data=grid_data,
        model_type=checkpoint["model_type"],
        n=checkpoint["n"],
        top_p=checkpoint["top_p"],
        val_timesteps=args.val_timesteps,
        device=device,
        dataset=DATASET,
        data_path=JHTDB_PATH,
        val_z=VAL_Z,
    )

# ALDD

To run the project install [uv](https://docs.astral.sh/uv/), then run:
```bash
git clone https://github.com/meditatoire/ALDD.git
cd ALDD
uv run ./src/training.py
```

Training saves checkpoints in `./models`, for example `models/fno_e30_k3_n16_top25.pt`.

To validate a saved checkpoint:
```bash
uv run ./src/validation.py models/fno_e30_k3_n16_top25.pt
```

For JHTDB, set the training planes in `src/training.py` with `TRAIN_Z` and run:
```bash
uv run ./src/training.py
```
For validation, set `DATASET`, `JHTDB_PATH`, and `VAL_Z` at the top of
`src/validation.py`, then give the checkpoint path as usual:
```bash
uv run ./src/validation.py models/fno_e30_k3_n16_top25.pt
```

Keep validation `z` values out of `TRAIN_Z`. Each plane is a separate 2D
trajectory; time pairs are created only within the same plane.

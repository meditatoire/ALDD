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

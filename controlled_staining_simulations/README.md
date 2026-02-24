This folder contains notebooks and scripts for running controlled staining simulations, extracting features, training ABMIL-style models, and evaluating results.

Contents
- `extract_features.py`: extract tile features from WSIs/tiles for downstream training or evaluation.
- `infere_simulated_models.py`: run inference on models trained with simulated staining.
- `infere_public_models.py`: run inference using public/foundation models.
- `train_abmil.py`: example script to train an ABMIL model.
- `evaluate_results.ipynb`: notebook for aggregating and plotting evaluation metrics.
- `simulation_settings.ipynb`: notebook describing simulation parameters and example runs.

Quick start
- Prepare data and model checkpoints as described in the repo root README.
- Run feature extraction (example):

  python extract_features.py --help

- Use the notebooks for interactive analysis and reproducing experiments.

Notes
- Scripts accept CLI args; open them or run `--help` to view options.
- Model checkpoint placement: see `../models/README.md` for expected paths.

Utility scripts used across the repository.

Contents
- `gpu_monitor.py`: helpers to monitor GPU usage.
- `load_encoder.py`: load pretrained encoders used for feature extraction.
- `tile_utils.py`: tile extraction and IO helpers.
- `tissue_detector.py`: simple tissue-detection utilities.
- `train_utils.py`: training helpers and common training loops.
- `utils_global.py`, `utils_stainUnmix.py`: assorted helpers used by multiple modules.

Usage
- Import the required helper from `utils` in your scripts or call the command-line utilities where provided.
- These modules are lightweight and meant to be called by the scripts/notebooks in the top-level folders.

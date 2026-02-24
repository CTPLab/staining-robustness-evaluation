
**A protocol for evaluating robustness to H&E staining variation in computational pathology models**

This repository contains code and examples for controlled staining simulations, stain unmixing, feature extraction, and model inference for evaluating computational pathology models under H&E staining variation.

**Repository Layout**
- **controlled_staining_simulations**: Notebooks and scripts to run staining simulations, train simulated models, and evaluate results. See [controlled_staining_simulations](controlled_staining_simulations/).
- **models**: Checkpoints and model wrappers (place downloaded model files here). See [models](models/).
- **stain_vector_concentration_extraction**: Tools and scripts to unmix stain vectors and extract concentrations from tiles/WSIs. See [stain_vector_concentration_extraction](stain_vector_concentration_extraction/).
- **utils**: Utility scripts for tile handling, GPU monitoring, model loading, and training helpers. See [utils](utils/).
- **SURGEN.csv, tcga_coadread.csv**: Datasets / metadata used in this project.

**Quick Start**
- Prerequisites: Python 3.8+ and a working PyTorch install (GPU recommended). Install common packages used across scripts: `pip install torch torchvision numpy scikit-image openslide-python tifffile pandas scikit-learn` or create a conda env and install equivalents.
- Notebooks: For end-to-end examples and recommended workflows, open the notebooks in [controlled_staining_simulations](controlled_staining_simulations/).
- Scripts: Example command (run from this repository root):

	python controlled_staining_simulations/extract_features.py

	Most scripts accept arguments; use `--help` on each script to see usage.

**Models (download and placement)**
- This repo expects model checkpoints to be downloaded separately. Create a folder under `models/` for each model and place the checkpoint files there.
- Example expected placements:
	- `models/NIEHEUS2023/` → put `export-0.pth`, `export-1.pth`, ...
	- `models/WAGNER2023/` → put `MSI_high_CRC_model.pth`
	- `models/CTRANSPATH/` → put `ctranspath.pth`

Model download links:
- NIEHEUS2023 (HuggingFace): [add URL here]
- WAGNER2023 (HuggingFace): [add URL here]
- Other foundation models: add their source repos/links here.


**How to run common tasks**
- Unmix stains / extract concentrations: run scripts in [stain_vector_concentration_extraction](stain_vector_concentration_extraction/).
- Extract tile features: use [controlled_staining_simulations/extract_features.py](controlled_staining_simulations/extract_features.py) or the notebook variants.
- Run inference on simulated or public models: use `controlled_staining_simulations/infere_simulated_models.py` and `controlled_staining_simulations/infere_public_models.py`.
- Evaluation: see `controlled_staining_simulations/evaluate_results.ipynb` for typical analysis and plotting of results.

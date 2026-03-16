# A protocol for evaluating robustness to H&E staining variation in computational pathology models

This repository implements a structured protocol for evaluating the robustness of computational pathology (CPath) models to hematoxylin and eosin (H&E) staining variation, as described in our paper [Link to Paper](https://arxiv.org/abs/2603.12886).

<img width="3309" height="1443" alt="Figure 1-git" src="https://github.com/user-attachments/assets/89f92af2-4399-428f-bdd8-fad460d05fb5" />

It enables:
- Definition of realistic reference staining conditions  
- Extraction of slide-level staining properties  
- Controlled simulation of staining variation during model application  
- Quantification of robustness via performance variability  

The protocol is demonstrated using MSI classification in colorectal cancer, but is designed to be reusable for other tasks and datasets.

---

## Overview: How This Project Is Structured

The project is split into this code repository and a HuggingFace repository [HuggingFace](https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation) providing the precomputed results.
You can reproduce everything or selectively reuse precomputed artifacts:

| Work Package | Code (This Repo) | Precomputed Results (HuggingFace) |
|---------------|---------------------|---------------------| 
| PLISM stain characterization | `stain_vector_concentration_extraction/compute_stats.py`, `stain_vector_concentration_extraction/unmix_tiles.py` | `plism-wsi_stain_references` |
| SurGen stain characterization | `stain_vector_concentration_extraction/unmix_wsi_v1.py` | `surgen_stain_properties` |
| Sample ABMIL train hyperparameters | `controlled_staining_simulations/simulation_settings.ipynb` | `MSI_classification_models/fixed_splits_n=300`, `MSI_classification_models/fixed_simulation_hps_n=300.csv` |
| ABMIL training (n=300 models) | `controlled_staining_simulations/train_abmil.py` | `MSI_classification_models/trained_models` |
| Extract features under simulated reference staining conditions | `controlled_staining_simulations/extract_features.py` | Not provided, follow steps in GitHub. |
| Apply models on extracted features | `controlled_staining_simulations/apply_simulated_models.py`, `controlled_staining_simulations/apply_public_models.py` | `exp_results` |
| Evaluate results | `controlled_staining_simulations/evaluate_results.ipynb` | See paper for results. |

The layout of this repository mirrors the protocol steps 1-3 (see Figure above).

### [stain_vector_concentration_extraction](stain_vector_concentration_extraction/)
Implements `Step 1: Staining reference selection` and `Step 2: Test set staining characterization`.

Estimates both H&E stain vectors and intensities by Macenko-based unmixing on the:
- Tile-level for PLISM dataset [More information on PLISM](https://p024eb.github.io/)
- Slide-level for SurGen or any other WSI-based dataset

Use this if you want to:
- Build your own stain reference library  
- Extract stain statistics from a new dataset  

### [controlled_staining_simulations](controlled_staining_simulations/)
Implements `Step 3: Apply CPath model under simulated reference conditions`, as well as the ABMIL-based training of the MSI classification models.

Provides code for:
- ABMIL-based training of MSI models
- Feature extraction under simulated staining conditions
- Apply public and self-trained models on features from simulated staining conditions
- Evaluation notebooks

Use this if you want to:
- Replicate our experiments (ABMIL-based training, feature extraction and model application)
- Run controlled staining simulations on your own models

### [models](models/)
Model wrappers and checkpoint storage.

Downloaded model weights must be placed here manually.

### [utils](utils/)
Shared utilities:
- Tile handling  
- GPU monitoring  
- Model loading  
- Training helpers  

### Metadata Files
Used for dataset splits and evaluation in the manuscript.
- `SURGEN.csv`
- `tcga_coadread.csv`

---

## Quick Start

### 1. Setup

- Python ≥ 3.8  
- PyTorch (GPU recommended)

Install core dependencies:

```bash
pip install torch torchvision numpy scikit-image openslide-python opencv-python-headless timm huggingface-hub einops pillow scipy shapely pandas matplotlib seaborn scikit-learn
```
Move to the code directory:
```bash
cd /YOUR/PATH/staining-robustness-evaluation
```

### 2. Typical Workflows

#### A. Apply the protocol to your own dataset

1. **Select reference staining conditions**
	- Option 1: Select references from our PLISM-based H&E stain vectors and intensity library [Hugging Face Repository](https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main)
	- Option 2: Create your own references based on your own WSIs (Script: [unmix_wsi_v1.py](stain_vector_concentration_extraction/unmix_wsi_v1.py)) or tiles (Script: [unmix_tiles.py](stain_vector_concentration_extraction/unmix_tiles.py)).

2. **Characterize test set staining properties**

Configure and run [unmix_wsi_v1.py](stain_vector_concentration_extraction/unmix_wsi_v1.py), check the unmixing logs to verify suitable tiles are selected, if the selected tiles are problematic, please adjust the threshold parameters in [config.py](stain_vector_concentration_extraction/config.py)

```bash
python -m stain_vector_concentration_extraction.unmix_wsi_v1
```
For detailed steps check: [stain_vector_concentration_extraction](stain_vector_concentration_extraction/)

3. **Apply CPath models under simulated reference staining conditions**

First, extract the features (utilizing your selected foundation model encoder) under the simulated staining condition, which is applied for each tile before feature extraction.
```bash
python -m controlled_staining_simulations.extract_features --help
```
Second, apply your trained aggregator model on the extracted features: 
```bash
python -m controlled_staining_simulations.apply_public_models --help
```
For detailed steps check [controlled_staining_simulations](controlled_staining_simulations/).

4. **Evaluate robustness**
You can build on our Jupyter Notebook for result evaluation, or implement your own custom logic for evaluating the results:
```
controlled_staining_simulations/evaluate_results.ipynb
```

---

#### B. Reproduce experiments from the paper

1. Download pretrained models (see section below) and place checkpoints under `models/`
2. Run feature extraction
3. Run model application scripts
4. Use the evaluation notebook to reproduce AUC and robustness metrics

For detailed steps check [controlled_staining_simulations](controlled_staining_simulations/).

---

## Models: Download and Placement

Create subfolders under `models/` and place checkpoints there.

Example structure:

```
models/NIEHEUS2023/export-0.pth
models/WAGNER2023/MSI_high_CRC_model.pth
models/CTRANSPATH/ctranspath.pth
...
```

Foundation models:
- UNI2-h: [HuggingFace](https://huggingface.co/MahmoodLab/UNI2-h)
- HOptimus1: [HuggingFace](https://huggingface.co/bioptimus/H-optimus-1)
- Virchow2: [HuggingFace](https://huggingface.co/paige-ai/Virchow2)
- CTransPath: [GitHub](https://github.com/Xiyue-Wang/TransPath?tab=readme-ov-file)
- RetCCL: [GitHub](https://github.com/Xiyue-Wang/RetCCL)

Public MSI models:
- NIEHEUS2023: [HuggingFace](https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main/MSI_classification_models/NIEHEUS2023), [Original Repo](https://github.com/KatherLab/crc-models-2022/tree/main/Quasar_models/Wang%2BattMIL/isMSIH), [Paper](https://www.sciencedirect.com/science/article/pii/S2666379123000861?via%3Dihub)
- WAGNER2023: [HuggingFace](https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main/MSI_classification_models/WAGNER2023), [Original Repo](https://github.com/peng-lab/HistoBistro/tree/main/CancerCellCRCTransformer/trained_models), [Paper](https://www.sciencedirect.com/science/article/pii/S1535610823002787?via%3Dihub)

**Data and pretrained models:** Pretrained models, stain reference libraries, and extracted stain statistics are available on the accompanying [Hugging Face Repository](https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main)

---

## Evaluation Metrics

- **Performance:** AUC under the reference staining condition  
- **Robustness:** Min–max AUC range across all simulated staining conditions  

Bootstrapped confidence intervals (n=1000) are computed in the evaluation notebook.

---

## Reproducibility and Resources

Resources available on Hugging Face:

- Stain reference library (PLISM)
- Slide-level stain vectors (SurGen)  
- Trained ABMIL models  
- Hyperparameter configurations  

👉 Hugging Face: [Hugging Face Repository](https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main)

---

## Citation

If you use this repository, please cite:
[A protocol for evaluating robustness to H&E staining variation in computational pathology models
](https://arxiv.org/abs/2603.12886)
```
@misc{schönpflug2026protocolevaluatingrobustnesshe,
      title={A protocol for evaluating robustness to H&E staining variation in computational pathology models}, 
      author={Lydia A. Schönpflug and Nikki van den Berg and Sonali Andani and Nanda Horeweg and Jurriaan Barkey Wolf and Tjalling Bosse and Viktor H. Koelzer and Maxime W. Lafarge},
      year={2026},
      eprint={2603.12886},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.12886}, 
}
```

## References
This repository utilizes and builds on:

**Datasets:**
* PLISM dataset: Ochi, M., Komura, D., Onoyama, T. et al. Registered multi-device/staining histology image dataset for domain-agnostic machine learning models. Sci Data 11, 330 (2024). [Link](https://doi.org/10.1038/s41597-024-03122-5)
* SurGen dataset: Myles C., Um, I.H., Marshall, C. et al. 1020 H&E-stained whole-slide images with survival and genetic markers. GigaScience, Volume 14 (2025). [Link](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf086/8277208?login=true)
* TCGA COADREAD: WSIs: [GDC Portal](https://portal.gdc.cancer.gov/), MSI Status from CBioportal: [TCGA COADREAD Pan-cancer Atlas (2018)](https://www.cbioportal.org/study/summary?id=coadread_tcga_pan_can_atlas_2018), [TCGA COADREAD Nature (2012)](https://www.cbioportal.org/study/summary?id=coadread_tcga_pub). The results shown here are in whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga.

**Foundation models:**
- UNI2-h: Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al. Towards a general-purpose foundation model for computational pathology. Nat Med (2024). [Paper](https://doi.org/10.1038/s41591-024-02857-3), [HuggingFace](https://huggingface.co/MahmoodLab/UNI2-h)
- HOptimus1: [HuggingFace](https://huggingface.co/bioptimus/H-optimus-1)
- Virchow2: Zimmermann, E., Vorontsov, E., Viret et al. Virchow2: Scaling self-supervised mixed magnification models in pathology (2024). [Paper](arXiv:2408.00738), [HuggingFace](https://huggingface.co/paige-ai/Virchow2)
- CTransPath: Wang, X., Yang, S., Zhang et. al. Transformer-based unsupervised contrastive learning for histopathological image classification. Medical image analysis, 81, p.102559 (2022). [Paper](https://www.sciencedirect.com/science/article/pii/S1361841522002043), [GitHub](https://github.com/Xiyue-Wang/TransPath?tab=readme-ov-file)
- RetCCL: Wang, X., Du, Y., Yang, S. et. al. RetCCL: Clustering-guided contrastive learning for whole-slide image retrieval. Medical image analysis, 83, 102645 (2023). [Paper](https://www.sciencedirect.com/science/article/pii/S1361841522002730), [GitHub](https://github.com/Xiyue-Wang/RetCCL)

**Public MSI models:**
- Niehues, J. M., Quirke, P., West, N. P., et. al. Generalizable biomarker prediction from cancer pathology slides with self-supervised deep learning: A retrospective multi-centric study. Cell reports medicine, 4(4) (2023). [Paper](https://www.sciencedirect.com/science/article/pii/S2666379123000861?via%3Dihub), [HuggingFace](https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main/MSI_classification_models/NIEHEUS2023), [Original Repo](https://github.com/KatherLab/crc-models-2022/tree/main/Quasar_models/Wang%2BattMIL/isMSIH)
- Wagner, S. J., Reisenbüchler, D., West, N. P. et al. Transformer-based biomarker prediction from colorectal cancer histology: A large-scale multicentric study. Cancer cell, 41(9), 1650-1661 (2023). [Paper](https://www.sciencedirect.com/science/article/pii/S1535610823002787?via%3Dihub), [HuggingFace](https://huggingface.co/datasets/CTPLab-DBE-UniBas/staining-robustness-evaluation/tree/main/MSI_classification_models/WAGNER2023), [Original Repo](https://github.com/peng-lab/HistoBistro/tree/main/CancerCellCRCTransformer/trained_models)

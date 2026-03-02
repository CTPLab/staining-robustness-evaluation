# Infere CPath models under simulated reference staining conditions

This repository supports two main use cases:

1. **Reproduce the experiments from the paper (MSI in CRC, SurGen cohort).**
2. **Apply the protocol to your own dataset and CPath models.**

The workflow follows the three protocol steps described in the manuscript.

---
## Repository Structure

| Protocol Step | Repository Component |
|---------------|---------------------|
| Reference stain selection | PLISM library (download from HuggingFace) |
| Test cohort stain characterization | `stain_vector_concentration_extraction/` |
| Controlled simulation + feature extraction | `extract_features.py` |
| Model inference | `infere_simulated_models.py`, `infere_public_models.py` |
| Training simulated ABMIL models | `train_abmil.py` |

---

## Applying the Protocol to Your Own Dataset

This assumes you already selected reference staining conditions and extracted H&E stain vectors and intensities for your own datasets. If this is not the case please check [stain_vector_concentration_extraction](stain_vector_concentration_extraction/) and [README](./README.md).

### 1. Extract Features Under Controlled Simulations
The script expects a csv file with `slide_id`, `slide_path` and `task` columns, where task should contain the (binary) labels.

e.g. utilizing Uni2-h: 
```bash
python extract_features.py \
    --task ##NAME-OF-YOUR-TASK-COLUMN## \
    --csv  ##PATH/TO/YOUR/DATASET/CSV## \
    --output_dir  ##PATH/TO/YOUR/OUTPUTDIR## \
    --foundation_models Univ2 \
    --stain_dir ##PATH/TO/YOUR/DATASETS/STAIN-PROPS## \
    --ref_stain_dir ##PATH/TO/YOUR/REFERENCE-LIBRARY## \ 
    --um_size 224 \ ## set as desired, currently 224px X 224px at 1 um/px (10x)
    --px_size 224 \ ## set as desired
    --gpu_id 0 \
    --batch_size 1000 \
    --visualize
```

Outputs:
```
features_output/
    intensity=None_stain=None/
        UNIv2_features_224um_224px_fcnn/
            SLIDE_ID.npz
```

Each `.npz` contains:
- `embeds`
- `coords`

Repeat this for:
- Reference condition
- All simulated staining conditions

---
### 2. Run Inference
I recommend building on `infere_public_models.py`, which expects features extracted with `extract_features.py` and allows providing a pre-trained model path. 

Please add your custom aggregator architecture in the script:
```python
        if "wagner2023" in pretrained_path.lower():
            foundation_model = "ctranspath"
            aggregator = Wagner2023(
                input_feature_size=INPUT_FEATURE_SIZE[foundation_model],
                n_classes=n_classes,
            )
        elif "niehues2023" in pretrained_path.lower():
            foundation_model = "retccl"
            aggregator = Niehues2023(
                input_feature_size=INPUT_FEATURE_SIZE[foundation_model],
                n_classes=n_classes,
            )
        else:
            raise ValueError(f"Unknown model name in pretrained path: {pretrained_path}")
```

Please repeat for each staining condition, setting `--features_dir features_output/intensity=None_stain=None` accordingly.

e.g. for low intensity condition
```bash
python controlled_staining_simulations/infere_simulated_models.py \
  --task ##NAME-OF-YOUR-TASK-COLUMN## \
  --csv  ##PATH/TO/YOUR/DATASET/CSV## \
  --features_dir ##PATH/TO/YOUR/FEATURESDIR##/intensity=KRH_GT450_stain=None \
  --pretrained_model ###PATH/TO/YOUR/PRETRAINED-MODEL ### \
  --output_dir results \
  --gpu_id 0
```

Outputs:
```
results/
    test_results_reference.csv
    predictions_exp_id=0_SURGEN_N738.csv
    ...
```
---
### Step 3: Evaluate results
Adapt the Jupyter Notebook [evaluate_results.ipynb](evaluate_results.ipynb) as needed, to evaluate your generated results.

---
## Reproducing the Paper Experiments

### Step 0 – Download Required Assets

Before running anything:

- Download **PLISM reference stain library** (reference stain vectors + intensities).
- Download **SurGen stain properties** (slide-specific stain vectors + intensities).
- Download **trained ABMIL models + simulation configs** (for simulated models).
- Download **public pretrained models** (e.g. Wagner2023, Niehues2023).
- Download desired **foundation models** (UNI2-h, H-Optimus-1, Virchow2, CTransPath, RetCCL).

Place:
- ABMIL checkpoints into `models/` or use `--sim_pretrained_dir`
- Public model weights into `models/`
- PLISM + SurGen assets in accessible folders (paths passed via CLI)

This assumes you already selected reference staining conditions and extracted H&E stain vectors and intensities for your own datasets. If this is not the case please check [stain_vector_concentration_extraction](stain_vector_concentration_extraction/) and [README](./README.md).

---

### Step 1 – Extract Features Under Controlled Staining
Four stain conditions:
| Condition                 | `--target_stain_name` | `--target_intensity` |
| ------------------------- | --------------------- | -------------------- |
| Reference                 | None                  | None                 |
| Low intensity             | None                  | KRH_GT450            |
| High intensity            | None                  | GV_AT2               |
| Low H&E color similarity  | HRH_S60               | None                 |
| High H&E color similarity | GV_GT450              | None                 |

Eg. for low intensity condition:

Uni2-h, HOptimus1, Virchow2  -> 224px X 224px at 1 um/px (10x): 
```bash
python extract_features.py \
    --task MSI \
    --csv SURGEN.csv \
    --output_dir ##PATH/TO/YOUR/OUTPUTDIR## \
    --foundation_models Univ2,HOptimus1,Virchow2 \
    --stain_dir ##PATH-TO-SURGEN-STAINS## \
    --ref_stain_dir ##PATH-TO-PLISM-REFERENCES## \ 
    --um_size 224 \
    --px_size 224 \
    --target_stain_name None \ 
    --target_intensity \
    --gpu_id 0 \
    --batch_size 1000 \
    --visualize
```

CTransPath, RetCCL -> 224px X 224px at 0.875 um/px (11.4x): 
```bash
python extract_features.py \
    --task MSI \
    --csv SURGEN.csv \
    --output_dir ##PATH/TO/YOUR/OUTPUTDIR## \
    --foundation_models Univ2,HOptimus1,Virchow2 \
    --stain_dir ##PATH/TO/surgen_stain_properties## \
    --ref_stain_dir ##PATH/TO/plism-wsi_stain_references## \ 
    --um_size 256 \
    --px_size 224 \
    --target_stain_name None \ 
    --target_intensity \
    --gpu_id 0 \
    --batch_size 1000 \
    --visualize
```

Outputs:
```
features/
    intensity=None_stain=None/
        UNIv2_features_224um_224px_fcnn/
            SLIDE_ID.npz
```

Each `.npz` contains:
- `embeds`
- `coords`

Repeat this for:
- Reference condition
- All simulated staining conditions

---

### Step 2A – Inference: Simulated ABMIL Models (n=300)

Runs inference for all 300 models for one staining condition. Please repeat for each staining condition, setting `--features_dir features_output/intensity=None_stain=None` accordingly.

e.g. for low intensity condition
```bash
python controlled_staining_simulations/infere_simulated_models.py \
  --csv SURGEN.csv \
  --features_dir features_output/intensity=KRH_GT450_stain=None \
  --sim_settings_csv ###PATH/TO/fixed_simulation_hps_n=300.csv### \
  --sim_pretrained_dir ###PATH/TO/trained_models### \
  --output_dir results \
  --gpu_id 0
```

Outputs:
```
results/
    test_results_reference.csv
    predictions_exp_id=0_SURGEN_N738.csv
    ...
```

---

### Step 2B – Inference: Public Models
Runs inference for a public model on one staining condition. Please repeat for both public models (WAGNER2023, NIEHEUS2023) and each staining condition, setting `--features_dir features_output/intensity=None_stain=None` accordingly.

```bash
python controlled_staining_simulations/infere_public_models.py \
  --csv SURGEN.csv \
  --feature_dir features_output/intensity=None_stain=None \
  --pretrained_model models/WAGNER2023/model.pth \
  --output_dir results \
  --gpu_id 0
```

Outputs:
```
results/
    test_results_reference.csv
    predictions_agg=WAGNER2023_SURGEN_N738.csv
```
### Step 4: Evaluate results
Follow the Jupyter Notebook [evaluate_results.ipynb](evaluate_results.ipynb) to evaluate the generated results, setting paths matching your local file paths.

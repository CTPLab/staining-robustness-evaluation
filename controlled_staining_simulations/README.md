# Apply CPath models under simulated reference staining conditions

This subfolder contains scripts to:
1. **Apply (your) CPath models on your own dataset and selected reference conditions.**
2. **Reproduce the experiments from the paper (MSI in CRC, SurGen cohort).**

---
## Folder Structure

| Protocol Step | Repository Component |
|---------------|---------------------|
| Controlled simulation + feature extraction | `extract_features.py` |
| Model application | `apply_simulated_models.py`, `apply_public_models.py` |
| Training simulated ABMIL models | `train_abmil.py` |

---

## Apply CPath models on your own dataset and selected reference conditions.

This assumes you already selected reference staining conditions and extracted H&E stain vectors and intensities for your own datasets. If this is not the case please check [stain_vector_concentration_extraction](stain_vector_concentration_extraction/) and [README](./README.md).

### 1. Extract Features Under Controlled Simulations
The script expects a csv file with `slide_id`, `slide_path` and `task` columns, where task should contain the (binary) labels.

e.g. utilizing Uni2-h: 
```bash
python extract_features.py \
    --task ##NAME-OF-YOUR-TASK-COLUMN## \
    --csv  ##PATH/TO/YOUR/DATASET/CSV## \
    --output_dir  ##PATH/TO/YOUR/OUTPUTDIR## \
    --foundation_models univ2 \
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
        univ2_features_224um_224px_fcnn/
            SLIDE_ID.npz
```

Each `.npz` contains:
- `embeds`
- `coords`

Repeat this for:
- Reference condition
- All simulated staining conditions

---
### 2. Run custom models on extracted features
I recommend building on `apply_public_models.py`, which expects features extracted with `extract_features.py` and allows providing a pre-trained model path. 

Please add your custom aggregator architecture in the script:
```python
        if "wagner2023" in pretrained_path.lower():
            foundation_model = "ctranspath"
            model_family = "wagner2023"
            aggregator = Wagner2023(pretrained_path=pretrained_path)
        elif "niehues2023" in pretrained_path.lower():
            foundation_model = "retccl"
            model_family = "niehues2023"
            aggregator = Niehues2023(pretrained_path=pretrained_path)
        ## FIXME: Add your own model loading logic here
        else:
            raise ValueError(f"Unknown model name in pretrained path: {pretrained_path}")
```

Please repeat for each staining condition, setting `--features_dir features_output/intensity=None_stain=None` accordingly.

e.g. for low intensity condition
```bash
python -m controlled_staining_simulations.apply_simulated_models \
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

`univ2`, `hoptimus1`, `virchow2` -> 224px X 224px at 1 um/px (10x): 
```bash
python -m controlled_staining_simulations.extract_features \
    --task MSI \
    --csv SURGEN.csv \
    --output_dir ##PATH/TO/YOUR/OUTPUTDIR## \
    --foundation_models univ2,hoptimus1,virchow2 \
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
python -m controlled_staining_simulations.extract_features \
    --task MSI \
    --csv SURGEN.csv \
    --output_dir ##PATH/TO/YOUR/OUTPUTDIR## \
    --foundation_models univ2,hoptimus1,virchow2 \
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
        univ2_features_224um_224px_fcnn/
            SLIDE_ID.npz
```

Each `.npz` contains:
- `embeds`
- `coords`

Repeat this for:
- Reference condition
- All simulated staining conditions

---

### Step 2A – Run simulated ABMIL Models (n=300)

Applies all 300 models for one staining condition. Please repeat for each staining condition, setting `--features_dir features_output/intensity=None_stain=None` accordingly.

e.g. for low intensity condition
```bash
python -m controlled_staining_simulations.apply_simulated_models \
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

### Step 2B – Run Public Models
Runs a public model on one staining condition. Please repeat for both public models (WAGNER2023, NIEHEUS2023) and each staining condition, setting `--features_dir features_output/intensity=None_stain=None` accordingly.

```bash
python -m controlled_staining_simulations.apply_public_models \
  --csv SURGEN.csv \
  --features_dir features_output/intensity=None_stain=None \
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

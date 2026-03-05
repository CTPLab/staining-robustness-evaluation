# Stain Vector & Intensity Extraction

This subfolder contains scripts to:
1. **Extract stain properties of your own dataset.**
2. **Reproduce the experiments from the paper (create PLISM staining library, extract SurGen stain characteristics).**

---
## Folder Structure

| Protocol Step | Repository Component |
|---------------|---------------------|
| PLISM stain characterization |  `compute_stats.py`, `unmix_tiles.py`|
| Test cohort stain characterization | `unmix_wsi_v1.py` |

---
## 1. Applying to Your Own Dataset

### WSI-Level Stain Extraction
Set configuration inside the script:
```python
### CONFIGURATION ###
num_tiles = 10  # <--- Number of valid tiles to extract per WSI
redo: bool = False  # <--- Force re-extraction of tiles
tissue_masks_dir = None  # <--- Path to precomputed tissue masks, script generates masks if not provided
csv = "YOUR-CSV.csv"  # <--- Path to your datasets CSV file, should contain slide_path and slide_id column
output_data_dir = f"./stain_vectors"  # <--- Path to store extracted stain vectors and intensities
output_report_dir = f"./logs"  # <--- Path to store analysis report images for each tile
output_intensity_dir = f"./intensities"  # <--- Path to store extracted stain intensities

## Defines tile size for selecting tiles from each WSI, default: 448px X 448px at 0.5um/px (20x)
tile_size = 224  # um
out_px = 448  # px
######################
```
Run script:
```bash
python -m stain_vector_concentration_extraction.unmix_wsi_v1
```

Outputs:
* `stain_vectors/<slide_id>.npz` – Slide-level median stain vectors and intensities
* `logs/<slide_id>/` – Visual analysis reports
  
**Notes**:
* Adjust `thresholds` in `config.py` for your dataset’s staining quality
* Tune `angular_percentile` and `angular_shift` if unmixing is not optimal
---

### Tips for Parameter Tuning
In case the visual analysis reports look off, either because (1) bad tiles are selected (e.g. lots of artifacts/ blood), or (2) unmixing failed/ looks problematic, please adjust the parameters in `config.py`. 

Here are some suggestions:
| Parameter                             | Effect                   | Suggested Action                                         |
| ------------------------------------- | ------------------------ | -------------------------------------------------------- |
| `frac_sat`, `entropy`, `colorfulness` | Tile selection quality   | Increase to exclude low-quality tiles                    |
| `lap_var`                             | Avoid blurry tiles       | Increase for stricter selection                          |
| `angular_percentile`                  | Stain separation         | Lower → stronger separation, higher → reduce artifacts   |
| `angular_shift`                       | Fine-tune H/E separation | Small positive/negative values to correct swapped colors |

---

## 2. Reproducing Our Experiments
### Step 0: Prerequisites

Before running any scripts, ensure the following data is available:

| Dataset | Required Files |
|---------|----------------|
| **PLISM-wsi** | Tiles in PNG format organized by `<stain>/<device>/tile.png` and CSV metadata file (`PLISM_wsi_en.csv`), [Dataset Download Link](https://plus.figshare.com/articles/dataset/Pathology_Images_of_Scanners_and_Mobilephones_PLISM_-_Whole_Slide_Images_Dataset/23614422) |
| **SurGen WSIs** | Whole Slide Images (WSIs) and CSV metadata file (`SURGEN.csv`) [Dataset Download Link](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1285) |

We converted the SurGen WSIs from the .czi into a .tif format, please contact us on this repo/ mail contact provided in the paper if you want to use the .tif version of the files, we are happy to provide them to you.

---
### Step 1: Compute Tile Metrics (PLISM)
Set configuration inside the script:
```python
### CONFIGURATION ###
data_dir = ""  # <--- Path to PLISM-wsi dataset
stats_dir = ""  # <--- Path to store computed img metrics
#####################
```
Run script:
```bash
python -m stain_vector_concentration_extraction.compute_stats
````
Outputs
* per-stain/device CSV metrics: `metrics_<stain>_<device>.csv`
---

### Step 2: Tile-Level Stain Extraction (PLISM)
Set configuration inside the script:
```python
### CONFIGURATION ###
data_dir = ""  # <--- Path to PLISM-wsi dataset
stats_dir = ""  # <--- Path to pre-computed img metrics
output_dir = ""  # <--- Path to store unmixing results and analysis reports
redo: bool = False  # <--- Force re-processing of tiles
#####################
```
Run script:
```bash
python -m stain_vector_concentration_extraction.unmix_tiles
```

Outputs:
* Stain vectors (`npz/`)
* Visual analysis reports (`analysis_reports/`)
---

### Step 3: WSI-Level Stain Extraction (SurGen)
Set configuration inside the script:
```python
### CONFIGURATION ###
num_tiles = 10  # <--- Number of valid tiles to extract per WSI
redo: bool = False  # <--- Force re-extraction of tiles
tissue_masks_dir = None  # <--- Path to precomputed tissue masks, script generates masks if not provided
csv = "../SURGEN.csv"  # <--- Path to SurGen CSV file
output_data_dir = f"./stain_vectors"  # <--- Path to store extracted stain vectors and intensities
output_report_dir = f"./logs"  # <--- Path to store analysis report images for each tile
output_intensity_dir = f"./intensities"  # <--- Path to store extracted stain intensities
tile_size = 224  # um
out_px = 448  # px
######################
```
Run script:
```bash
python -m stain_vector_concentration_extraction.unmix_wsi_v1
```

Outputs:
* `stain_vectors/<slide_id>.npz` – Slide-level median stain vectors and intensities
* `logs/<slide_id>/` – Visual analysis reports

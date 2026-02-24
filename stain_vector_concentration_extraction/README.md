Tools to unmix stains and extract stain-vector concentrations from tiles and WSIs.

Contents
- `unmix_tiles.py`: Unmix PLISM image tiles to estimate stain concentrations per tile.
- `unmix_wsi_v1.py`: Unmix whole-slide images (WSI) pipeline (tile extraction + unmixing).
- `compute_stats.py`: Compute PLISM image properties (e.g. entropy, blur, hues etc) for the PLISM dataset, used for selecting suitable tiles for extracting stain vectors and intensities.
- `config.py`: configuration defaults for unmixing and tile handling.

Quick start
- Run tile unmixing:

  python unmix_tiles.py --help

- For WSI processing, use `unmix_wsi_v1.py` and ensure OpenSlide and TIFF dependencies are installed.
- If unmixing fails for some WSIs, try adjusting thresholding or unmixing parameters in `config.py` 

Notes
- Outputs are per-tile concentration arrays (or CSV summaries) consumed by feature extraction and downstream analysis.
- See `../controlled_staining_simulations` notebooks for examples combining unmixing and evaluation.

This folder holds model definitions and checkpoints used by the project. Checkpoints are not included and must be downloaded separately.

Structure
- Python model wrappers: `abmil.py`, `nature_net.py`, `niehues2023.py`, `resnet50.py`, `wagner2023.py`.
- Subfolders for checkpoints: place downloaded checkpoint files under appropriately named subfolders (examples below).

Expected placements (examples)
- `models/NIEHEUS2023/` — place `export-0.pth`, `export-1.pth`, ...
- `models/WAGNER2023/` — place `MSI_high_CRC_model.pth`
- `models/CTRANSPATH/` — place `ctranspath.pth`
- `models/RetCCL/` — place `best_ckpt.pth`

Notes
- NIEHEUS2023 and WAGNER2023 are available from a HuggingFace repo connected to this project (links to be added).
- Other foundation models must be downloaded from their source repositories; add URLs in the root README.
- If filenames differ, either rename files to match the expected names above or update script CLI args to point to actual checkpoint paths.

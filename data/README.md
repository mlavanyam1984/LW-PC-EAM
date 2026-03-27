# data/ — Dataset Placeholder

This directory is intentionally empty in the repository.

The MVTec AD dataset (~5 GB) is NOT included because:
1. File size exceeds GitHub limits
2. Dataset has its own CC BY-NC-SA 4.0 non-commercial license

## Download

Option 1 — Kaggle website:
  https://www.kaggle.com/datasets/avdvhh/mvtec-defect-detection-dataset

Option 2 — Kaggle API:
  pip install kaggle
  python scripts/train_eval.py --download --data_root ./data

Extract so the structure is:
  data/mvtec_ad/bottle/train/good/
  data/mvtec_ad/bottle/test/broken_large/
  ... (15 categories total)

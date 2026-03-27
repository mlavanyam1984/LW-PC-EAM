# LW-PC-EAM: Attention-Driven Explainable PatchCore for Real-Time Anomaly Detection on Resource-Constrained Edge Devices

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-MVTec%20AD-red)](https://www.kaggle.com/datasets/avdvhh/mvtec-defect-detection-dataset)

> **Official implementation** of the paper:  
> *"Attention-Driven Explainable PatchCore for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"*

---

## Overview

**LW-PC-EAM** is a lightweight, edge-deployable industrial anomaly detection framework that combines:

- 🔬 **MobileNetV2** backbone for efficient patch-level feature extraction  
- 🧠 **Memory-conditioned attention** (4 heads, 128-dim Q/K) — directly coupled to anomaly scoring (no post-hoc gradients)  
- 📦 **PatchCore memory bank** with greedy minimax coreset sampling (10% retention)  
- 🗺️ **Explainable localization maps** via logistic squashing — generated in a single forward pass  


## Project Structure

```
LW-PC-EAM/
├── src/
│   ├── model.py            # LW-PC-EAM model (backbone, attention, memory bank, heatmap)
│   ├── dataset.py          # MVTec AD dataset loader + Kaggle auto-download
│   ├── metrics.py          # AUROC, F1, explainability metrics (LF, AS, CI), latency
│   ├── visualization.py    # Heatmap overlays, edge detection, plots
│   └── __init__.py
├── scripts/
│   ├── train_eval.py       # Main training + evaluation pipeline
│   ├── inference_demo.py   # Single-image inference demo
│   └── ablation_study.py   # Ablation experiments (Section 4.19)
├── configs/
│   └── config.yaml         # All hyperparameters
├── results/
│   ├── figures/            # Pre-generated sample output images
│   └── metrics.json        # Sample evaluation metrics
├── tests/
│   └── test_model.py       # Unit tests
├── data/
│   └── .gitkeep            # Placeholder (dataset downloaded separately)
├── requirements.txt
├── setup.py
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## Dataset

This project uses the **MVTec Anomaly Detection (MVTec AD)** dataset:

- **15 industrial categories** (bottle, carpet, leather, screw, etc.)
- **5,354 high-resolution images**
- **1,888 annotated anomaly regions** across **73 defect types**
- Train set: defect-free images only
- Test set: defect-free + anomalous images with ground-truth masks

> ⚠️ **The dataset is NOT included in this repository** due to its large size (~5 GB).  
> Please download it separately using one of the methods below.

### Download Options

**Option 1 — Automatic (Kaggle API):**
```bash
# Install Kaggle CLI and set up credentials (~/.kaggle/kaggle.json)
pip install kaggle
python scripts/train_eval.py --download --data_root ./data
```

**Option 2 — Manual (Kaggle website):**
1. Visit: https://www.kaggle.com/datasets/avdvhh/mvtec-defect-detection-dataset
2. Click **Download** (requires free Kaggle account)
3. Extract to `./data/mvtec_ad/`

**Option 3 — Official MVTec source:**
```
https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
```

### Expected Folder Structure After Download
```
data/
└── mvtec_ad/
    ├── bottle/
    │   ├── train/
    │   │   └── good/          ← 209 defect-free training images
    │   └── test/
    │       ├── good/          ← normal test images
    │       ├── broken_large/  ← anomalous test images
    │       └── ...
    ├── carpet/
    ├── leather/
    └── ... (15 categories total)
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/LW-PC-EAM.git
cd LW-PC-EAM

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate.bat     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**GPU (recommended for faster feature extraction):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

### 1. Demo (no dataset needed)
```bash
python scripts/inference_demo.py --demo
```
Runs inference on synthetic images and saves 4-panel visualizations to `outputs/demo/`.

### 2. Train on one MVTec category
```bash
python scripts/train_eval.py \
    --data_root ./data/mvtec_ad \
    --category bottle
```

### 3. Train on all 15 categories
```bash
python scripts/train_eval.py \
    --data_root ./data/mvtec_ad \
    --all_categories
```

### 4. Single-image inference
```bash
python scripts/inference_demo.py \
    --image_path ./data/mvtec_ad/bottle/test/broken_large/000.png \
    --model_path  ./outputs/bottle/memory_bank.pt
```

### 5. Ablation study
```bash
python scripts/ablation_study.py \
    --data_root ./data/mvtec_ad \
    --category bottle
```

---

### Hyperparameters (from paper Section 4.1)

| Parameter | Value |
|-----------|-------|
| Backbone | MobileNetV2 (ImageNet pretrained) |
| Input size | 224 × 224 RGB |
| Embedding dim | 512 |
| Attention heads | 4 |
| Q/K projection dim | 128 |
| Coreset ratio | 10% (greedy minimax) |
| Optimizer | Adam (lr=0.001, wd=1e-5) |
| Batch size | 32 |
| Quantization | INT8 (TensorRT / ONNX Runtime) |
| Latency benchmark | 500 runs (after 50 warm-up) |

---

## Evaluation Metrics

All metrics from the paper are implemented in `src/metrics.py`:

- **Detection:** AUROC, F1-Score, Precision, Recall, Detection Accuracy (Eq. 15)
- **Explainability:** Localization Fidelity (LF), Attribution Stability (AS), Clarity Index (CI), Explainability Score (Eq. 14)
- **Efficiency:** Inference Speed/FPS (Eq. 16), Reduced Latency (Eq. 17), Processing Time (Eq. 19)
- **Quality:** Similarity Score (Eq. 18), Reconstruction Error (Eq. 20)
- **Cost:** Operational Cost (Eq. 13, weights: β=0.45, γ=0.30, δ=0.25)

---

## Edge Deployment

The model targets two edge platforms:

| Platform | Quantization | Latency | FPS |
|----------|-------------|---------|-----|
| NVIDIA Jetson Xavier NX | INT8 (TensorRT) | **15 ms** | **68** |
| Raspberry Pi 4 | INT8 (ONNX Runtime) | ~40 ms | ~25 |
| Standard CPU (FP32) | None | ~70 ms | ~14 |

---

## Reproducibility

All experiments follow the official MVTec AD train/test split. Results are reported as **mean ± std** across 5 independent runs with 95% confidence intervals. Statistical significance is verified via paired t-tests (p < 0.05 against baselines).

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lwpceam2025,
  title   = {Attention-Driven Explainable PatchCore for Real-Time Anomaly Detection
             on Resource-Constrained Edge Devices},
  journal = {<Journal Name>},
  year    = {2025}
}
```

**Dataset citation:**
```bibtex
@misc{mvtec_kaggle_2019,
  author    = {{MVTec Software GmbH}},
  title     = {MVTec Defect Detection Dataset},
  year      = {2019},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/avdvhh/mvtec-defect-detection-dataset}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The MVTec AD dataset is subject to its own license. Please review the terms at:  
https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## Acknowledgements

- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) — Bergmann et al., CVPR 2019
- [PatchCore](https://arxiv.org/abs/2106.08265) — Roth et al., CVPR 2022
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Sandler et al., CVPR 2018

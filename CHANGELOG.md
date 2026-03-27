# Changelog

All notable changes to LW-PC-EAM are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] — 2025

### Added
- Initial public release accompanying the journal submission
- `LightweightBackbone`: MobileNetV2-based patch feature extractor (512-dim, 224×224 input)
- `MemoryConditionedAttention`: 4-head attention module with 128-dim Q/K projections, conditioned on coreset residuals
- `PatchCoreMemoryBank`: greedy minimax coreset sampling (10% retention ratio)
- `ExplainableLocalizationMap`: logistic squashing heatmap generation (Eq. 1–2), single forward pass
- Full training pipeline for all 15 MVTec AD categories
- Ablation study script (attention, coreset ratio, embedding dim, INT8 quantization)
- Latency benchmark (500-run timed inference, 50 warm-up)
- Explainability metrics: Localization Fidelity (LF), Attribution Stability (AS), Clarity Index (CI)
- All 20 equations from the paper implemented in `src/metrics.py`
- Unit test suite with 30+ test cases
- GitHub Actions CI workflow (Python 3.8–3.11)
- Pre-generated result figures for all categories
- Demo mode (synthetic images, no dataset required)
- Dataset auto-download via Kaggle API

"""
LW-PC-EAM: Lightweight PatchCore with Explainable Attention Mechanism
"""
from .model import LWPCEAM, LightweightBackbone, MemoryConditionedAttention, PatchCoreMemoryBank, ExplainableLocalizationMap
from .dataset import MVTecDataset, get_dataloaders, MVTEC_CATEGORIES
from .metrics import (
    compute_auroc,
    compute_precision_recall_f1,
    compute_detection_accuracy,
    explainability_score,
    LatencyBenchmark,
    evaluate_model,
)
from .visualization import visualize_prediction, heatmap_overlay, apply_edge_detection

__version__ = "1.0.0"
__all__ = [
    "LWPCEAM",
    "LightweightBackbone",
    "MemoryConditionedAttention",
    "PatchCoreMemoryBank",
    "ExplainableLocalizationMap",
    "MVTecDataset",
    "get_dataloaders",
    "MVTEC_CATEGORIES",
    "compute_auroc",
    "compute_precision_recall_f1",
    "compute_detection_accuracy",
    "explainability_score",
    "LatencyBenchmark",
    "evaluate_model",
    "visualize_prediction",
    "heatmap_overlay",
    "apply_edge_detection",
]

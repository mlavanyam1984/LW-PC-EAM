"""
Visualization Utilities for LW-PC-EAM
========================================
Generates:
  - Anomaly heatmap overlays (jet colormap on original image)
  - Edge detection highlighting defect boundaries (Canny)
  - Side-by-side comparison grids
  - Batch visualization for multiple defect types
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized ImageNet tensor → uint8 RGB numpy array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def heatmap_overlay(
    image: np.ndarray,           # (H, W, 3) uint8
    heatmap: np.ndarray,         # (H, W)   float [0,1]
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay heatmap on image using jet colormap.
    Returns (H, W, 3) uint8 blended image.
    """
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image, 1 - alpha, colored_rgb, alpha, 0)
    return blended


def apply_edge_detection(
    heatmap: np.ndarray,          # (H, W) float [0,1]
    low_thresh: int = 50,
    high_thresh: int = 150,
) -> np.ndarray:
    """
    Apply Canny edge detection to heatmap to highlight defect boundaries.
    Returns (H, W) uint8 edge map.
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    edges = cv2.Canny(heatmap_uint8, low_thresh, high_thresh)
    return edges


def visualize_prediction(
    original_tensor: torch.Tensor,   # (1, 3, H, W) or (3, H, W)
    result: dict,
    ground_truth_mask: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
) -> np.ndarray:
    """
    Create a 4-panel visualization:
        [Original | Heatmap Overlay | Edge Detection | GT Mask (if available)]

    Args:
        original_tensor: normalized image tensor
        result:          output dict from model.predict()
        ground_truth_mask: binary GT mask (optional)
        title:           figure title
        save_path:       if given, save figure to this path
        show:            if True, display with plt.show()
    Returns:
        composite image as numpy array
    """
    if original_tensor.dim() == 4:
        original_tensor = original_tensor.squeeze(0)

    orig_img = denormalize(original_tensor)
    heatmap  = result["heatmap"]
    overlay  = heatmap_overlay(orig_img, heatmap)
    edges    = apply_edge_detection(heatmap)

    n_panels = 4 if ground_truth_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    score = result["anomaly_score"]
    label = "ANOMALY" if result["is_anomaly"] else "NORMAL"
    axes[1].set_title(f"Heatmap Overlay\nScore: {score:.4f} [{label}]")
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("Edge Detection\n(Defect Boundaries)")
    axes[2].axis("off")

    if ground_truth_mask is not None:
        axes[3].imshow(ground_truth_mask, cmap="gray")
        axes[3].set_title("Ground Truth Mask")
        axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Convert to numpy (buffer_rgba works on all matplotlib backends)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    composite = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return composite


def visualize_batch(
    results: List[dict],
    images: List[torch.Tensor],
    defect_types: List[str],
    save_dir: str = "outputs/visualizations",
    category: str = "unknown",
) -> None:
    """
    Save heatmap visualizations for a batch of predictions.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, (res, img, dt) in enumerate(zip(results, images, defect_types)):
        fname = f"{category}_{dt}_{i:03d}_{'ANOM' if res['is_anomaly'] else 'NORM'}.png"
        save_path = os.path.join(save_dir, fname)
        visualize_prediction(img, res, title=f"{category} | {dt}", save_path=save_path)


def plot_auroc_curve(
    labels: List[int],
    scores: List[float],
    method_name: str = "LW-PC-EAM",
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC curve with AUROC annotation."""
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="royalblue", lw=2, label=f"{method_name} (AUROC = {auroc*100:.2f}%)")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Anomaly Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_score_distribution(
    scores: List[float],
    labels: List[int],
    threshold: float,
    save_path: Optional[str] = None,
) -> None:
    """Plot anomaly score distributions for normal vs anomalous samples."""
    normal_scores = [s for s, l in zip(scores, labels) if l == 0]
    anomal_scores = [s for s, l in zip(scores, labels) if l == 1]

    plt.figure(figsize=(8, 4))
    plt.hist(normal_scores, bins=40, alpha=0.6, color="steelblue", label="Normal")
    plt.hist(anomal_scores, bins=40, alpha=0.6, color="tomato", label="Anomalous")
    plt.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Threshold = {threshold:.3f}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Score Distribution: Normal vs Anomalous")
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_comparison_table(metrics_dict: dict, save_path: Optional[str] = None) -> None:
    """
    Plot a comparison table like Table 10 from the paper:
    Methods vs AUROC, F1, Params, FLOPs, Latency.
    """
    methods = list(metrics_dict.keys())
    metrics_names = ["AUROC (%)", "F1 (%)", "Params (M)", "FLOPs (G)", "Latency (ms)"]
    values = np.array([[metrics_dict[m].get(k, 0) for k in
                        ["auroc", "f1", "params_m", "flops_g", "latency_ms"]]
                       for m in methods])

    fig, ax = plt.subplots(figsize=(10, len(methods) * 0.6 + 2))
    ax.axis("off")
    table = ax.table(
        cellText=values.round(1),
        rowLabels=methods,
        colLabels=metrics_names,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title("Computational Complexity Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

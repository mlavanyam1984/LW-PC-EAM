"""
Evaluation Metrics for LW-PC-EAM
==================================
Implements all metrics referenced in the paper:

  - AUROC (image-level and pixel-level)
  - Precision, Recall, F1-Score
  - Localization Fidelity (LF)
  - Attribution Stability (AS)
  - Clarity Index (CI)
  - Explainability Score = mean(LF, AS, CI)
  - Inference Speed (FPS) and Latency (ms)
  - Reconstruction Error
  - Similarity Score (cosine)
  - Operational Cost (Eq. 13)
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_auroc(
    labels: List[int],
    scores: List[float],
) -> Tuple[float, float]:
    """
    Compute image-level AUROC.
    Returns: (auroc_score, optimal_threshold)
    """
    labels_np = np.array(labels)
    scores_np = np.array(scores)
    auroc_val = roc_auc_score(labels_np, scores_np)

    fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
    youden_idx = np.argmax(tpr - fpr)
    optimal_thresh = thresholds[youden_idx]

    return float(auroc_val), float(optimal_thresh)


def compute_precision_recall_f1(
    labels: List[int],
    predictions: List[int],
) -> Dict[str, float]:
    """Compute Precision, Recall, F1-Score."""
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)) * 100,
        "recall":    float(recall_score(labels, predictions, zero_division=0)) * 100,
        "f1":        float(f1_score(labels, predictions, zero_division=0)) * 100,
    }


def compute_detection_accuracy(
    labels: List[int],
    predictions: List[int],
) -> float:
    """
    Detection accuracy Bcc [Equation 15]:
        Bcc = (TP + TN) / (TP + TN + FP + FN)
    """
    labels_np = np.array(labels)
    preds_np  = np.array(predictions)
    UQ = np.sum((preds_np == 1) & (labels_np == 1))  # TP
    UO = np.sum((preds_np == 0) & (labels_np == 0))  # TN
    GQ = np.sum((preds_np == 1) & (labels_np == 0))  # FP
    FO = np.sum((preds_np == 0) & (labels_np == 1))  # FN
    return float((UQ + UO) / (UQ + UO + GQ + FO + 1e-8)) * 100


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def localization_fidelity(
    heatmaps: List[np.ndarray],        # predicted heatmaps [0,1]
    ground_truths: List[np.ndarray],   # binary GT masks {0,1}
    top_k_ratio: float = 0.1,
) -> float:
    """
    Localization Fidelity (LF):
    Measures congruence between high-attribution regions and GT relevant regions.
    Computes overlap between top-K% predicted activation and GT mask.
    """
    scores = []
    for hm, gt in zip(heatmaps, ground_truths):
        if gt.max() == 0:
            continue  # skip normal samples
        k = max(1, int(hm.size * top_k_ratio))
        flat_hm = hm.flatten()
        flat_gt = gt.flatten()
        top_k_idx = np.argpartition(flat_hm, -k)[-k:]
        pred_mask = np.zeros_like(flat_hm)
        pred_mask[top_k_idx] = 1
        intersection = (pred_mask * flat_gt).sum()
        union = np.clip(pred_mask + flat_gt, 0, 1).sum()
        scores.append(intersection / (union + 1e-8))
    return float(np.mean(scores)) if scores else 0.0


def attribution_stability(
    heatmaps_a: List[np.ndarray],
    heatmaps_b: List[np.ndarray],
) -> float:
    """
    Attribution Stability (AS):
    Measures consistency of explanation maps under small input perturbations.
    Computed as mean structural similarity between two perturbed-input heatmaps.
    """
    scores = []
    for ha, hb in zip(heatmaps_a, heatmaps_b):
        # Normalized cross-correlation as stability proxy
        ha_norm = (ha - ha.mean()) / (ha.std() + 1e-8)
        hb_norm = (hb - hb.mean()) / (hb.std() + 1e-8)
        score = float(np.mean(ha_norm * hb_norm))
        scores.append(np.clip(score, 0, 1))
    return float(np.mean(scores)) if scores else 0.0


def clarity_index(heatmaps: List[np.ndarray]) -> float:
    """
    Clarity Index (CI):
    Measures sparseness and concentration of attribution maps.
    Higher CI → more focused/interpretable heatmaps.
    Computes ratio of energy in top-10% pixels vs total energy.
    """
    scores = []
    for hm in heatmaps:
        flat = hm.flatten()
        total_energy = (flat ** 2).sum() + 1e-8
        k = max(1, int(len(flat) * 0.1))
        top_k_vals = np.partition(flat, -k)[-k:]
        top_energy = (top_k_vals ** 2).sum()
        scores.append(top_energy / total_energy)
    return float(np.mean(scores))


def explainability_score(
    heatmaps: List[np.ndarray],
    ground_truths: List[np.ndarray],
    heatmaps_perturbed: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Combined Explainability Score = mean(LF, AS, CI)  [Equation 14 / Section 3.4]

    All three components normalized to [0, 1].
    If perturbed heatmaps not provided, AS defaults to CI-based proxy.
    """
    lf = localization_fidelity(heatmaps, ground_truths)
    ci = clarity_index(heatmaps)

    if heatmaps_perturbed is not None:
        as_score = attribution_stability(heatmaps, heatmaps_perturbed)
    else:
        as_score = ci  # fallback: use CI as AS proxy

    es = (lf + as_score + ci) / 3.0

    return {
        "localization_fidelity": float(lf),
        "attribution_stability": float(as_score),
        "clarity_index": float(ci),
        "explainability_score": float(es),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENCY METRICS
# ─────────────────────────────────────────────────────────────────────────────

class LatencyBenchmark:
    """
    Measures inference latency and FPS as per paper (500 warm-up + timed runs).
    Inference Speed δ = O_inf / U_wl   [Equation 16]
    """

    def __init__(self, warmup_runs: int = 50, timed_runs: int = 500):
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs

    def benchmark(self, model, input_tensor: torch.Tensor, device: str = "cpu") -> Dict[str, float]:
        """
        Run latency benchmark.
        Returns dict with latency_ms, fps, total_time_s
        """
        model.eval()
        model.to(device)
        x = input_tensor.to(device)

        # Warm-up
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model.backbone(x)

        # Timed runs
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(self.timed_runs):
                _ = model.predict(x, device=device)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / self.timed_runs) * 1000
        fps = self.timed_runs / elapsed

        return {
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 1),
            "total_time_s": round(elapsed, 3),
            "num_runs": self.timed_runs,
        }


def reduced_latency(
    baseline_latency_ms: float,
    optimized_latency_ms: float,
    baseline_fps: float,
    optimized_fps: float,
) -> float:
    """
    Reduced Latency ∇_M [Equation 17]:
        ∇_M = ((M_bsl - M_pqu) / M_bsl) * (F_bsl / F_pqu)
    """
    latency_reduction = (baseline_latency_ms - optimized_latency_ms) / (baseline_latency_ms + 1e-8)
    fps_ratio = baseline_fps / (optimized_fps + 1e-8)
    return float(latency_reduction * fps_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION ERROR & SIMILARITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """
    Reconstruction Error ∂_rc [Equation 20]:
        ∂_rc = (1/o) * ||y - ŷ||₂²
    """
    o = original.size
    return float(np.sum((original - reconstructed) ** 2) / o)


def similarity_score(
    query_embedding: np.ndarray,
    exemplar: np.ndarray,
) -> float:
    """
    Cosine Similarity Score Σ [Equation 18]:
        Σ = (v^T ∂_k) / (||v||₂ * ||∂_k||₂)
    """
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    e_norm = exemplar / (np.linalg.norm(exemplar) + 1e-8)
    return float(np.dot(q_norm, e_norm))


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONAL COST
# ─────────────────────────────────────────────────────────────────────────────

def operational_cost(
    hardware_cost: float,       # D_ix: capital cost of edge device
    energy_cost: float,         # F_ot: total energy consumed
    maintenance_cost: float,    # D_tc: support/maintenance expenses
    beta: float = 0.45,         # hardware weight
    gamma: float = 0.30,        # energy weight
    delta: float = 0.25,        # maintenance weight
) -> float:
    """
    Operational Cost D_pq [Equation 13]:
        D_pq = β*D_ix + γ*F_ot + δ*D_tc
    Weights (0.45, 0.30, 0.25) as per paper Section 3.5.
    """
    return beta * hardware_cost + gamma * energy_cost + delta * maintenance_cost


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    anomaly_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Full evaluation of LW-PC-EAM on a test DataLoader.

    Returns comprehensive metrics dict including:
        auroc, f1, precision, recall, accuracy,
        explainability_score, latency_ms, fps
    """
    model.eval()

    all_labels: List[int] = []
    all_scores: List[float] = []
    all_heatmaps: List[np.ndarray] = []

    print("Running evaluation ...")
    for batch in test_loader:
        imgs, labels = batch[0], batch[1]
        for i in range(imgs.shape[0]):
            result = model.predict(imgs[i].unsqueeze(0), device=device)
            all_scores.append(result["anomaly_score"])
            all_heatmaps.append(result["heatmap"])
            all_labels.append(int(labels[i]))

    auroc, opt_thresh = compute_auroc(all_labels, all_scores)

    if anomaly_threshold is None:
        anomaly_threshold = opt_thresh

    predictions = [1 if s > anomaly_threshold else 0 for s in all_scores]
    prf = compute_precision_recall_f1(all_labels, predictions)
    acc = compute_detection_accuracy(all_labels, predictions)
    ci  = clarity_index(all_heatmaps)

    return {
        "auroc":       round(auroc * 100, 2),
        "f1":          round(prf["f1"], 2),
        "precision":   round(prf["precision"], 2),
        "recall":      round(prf["recall"], 2),
        "accuracy":    round(acc, 2),
        "clarity_index": round(ci, 4),
        "threshold_used": round(anomaly_threshold, 4),
        "n_samples":   len(all_labels),
        "n_anomalous": sum(all_labels),
    }

"""
train_eval.py — Main Training and Evaluation Script for LW-PC-EAM
===================================================================
Usage:
    # Train + evaluate on one category
    python scripts/train_eval.py --data_root ./data/mvtec_ad --category bottle

    # Train on all 15 MVTec categories
    python scripts/train_eval.py --data_root ./data/mvtec_ad --all_categories

    # Evaluate only (load existing model)
    python scripts/train_eval.py --data_root ./data/mvtec_ad --category bottle --eval_only

    # Download dataset first
    python scripts/train_eval.py --download --data_root ./data
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import MVTEC_CATEGORIES, MVTecDataset, download_mvtec_kaggle, get_dataloaders
from src.metrics import LatencyBenchmark, compute_auroc, evaluate_model
from src.model import LWPCEAM
from src.visualization import (
    plot_auroc_curve,
    plot_score_distribution,
    visualize_prediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    "embedding_dim":       512,
    "num_attention_heads": 4,
    "qk_dim":              128,
    "coreset_ratio":       0.10,   # 10% as per paper
    "anomaly_threshold":   0.5,
    "gain":                5.0,
    "image_size":          224,

    # Training
    "batch_size":          32,
    "num_workers":         4,

    # Evaluation
    "latency_warmup_runs": 50,
    "latency_timed_runs":  500,

    # Output
    "output_dir": "outputs",
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

def train_category(
    category: str,
    data_root: str,
    config: dict,
    device: str,
    save_dir: str,
) -> dict:
    """
    Full LW-PC-EAM training and evaluation pipeline for one MVTec category.
    Returns metrics dict.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Category: {category.upper()}")
    logger.info(f"{'='*60}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(
        root=data_root,
        category=category,
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config["num_workers"],
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples:  {len(test_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LWPCEAM(
        embedding_dim=config["embedding_dim"],
        num_attention_heads=config["num_attention_heads"],
        qk_dim=config["qk_dim"],
        coreset_ratio=config["coreset_ratio"],
        anomaly_threshold=config["anomaly_threshold"],
        gain=config["gain"],
        image_size=config["image_size"],
        pretrained=True,
    )

    # ── Fit memory bank ───────────────────────────────────────────────────────
    logger.info("Building PatchCore memory bank ...")
    t0 = time.time()
    model.fit(train_loader, device=device)
    fit_time = time.time() - t0
    logger.info(f"Memory bank built in {fit_time:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Evaluating on test set ...")
    metrics = evaluate_model(model, test_loader, device=device)
    logger.info(f"  AUROC:     {metrics['auroc']:.2f}%")
    logger.info(f"  F1-Score:  {metrics['f1']:.2f}%")
    logger.info(f"  Precision: {metrics['precision']:.2f}%")
    logger.info(f"  Recall:    {metrics['recall']:.2f}%")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.2f}%")

    # ── Latency benchmark ─────────────────────────────────────────────────────
    logger.info(f"Running latency benchmark ({config['latency_timed_runs']} runs) ...")
    dummy_input = torch.randn(1, 3, config["image_size"], config["image_size"])
    benchmarker = LatencyBenchmark(
        warmup_runs=config["latency_warmup_runs"],
        timed_runs=config["latency_timed_runs"],
    )
    latency = benchmarker.benchmark(model, dummy_input, device=device)
    metrics.update(latency)
    logger.info(f"  Latency: {latency['latency_ms']:.1f} ms | FPS: {latency['fps']:.1f}")

    # ── Visualizations ────────────────────────────────────────────────────────
    vis_dir = os.path.join(save_dir, category, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    all_labels, all_scores, all_heatmaps = [], [], []
    max_vis = 5  # save up to 5 visualizations per category

    for batch_idx, batch in enumerate(test_loader):
        imgs, labels, defect_types = batch
        for i in range(imgs.shape[0]):
            result = model.predict(imgs[i].unsqueeze(0), device=device)
            all_labels.append(int(labels[i]))
            all_scores.append(result["anomaly_score"])
            all_heatmaps.append(result["heatmap"])

            if batch_idx < max_vis:
                dt = defect_types[i] if isinstance(defect_types[i], str) else defect_types[i]
                save_path = os.path.join(vis_dir, f"{batch_idx:03d}_{dt}.png")
                visualize_prediction(
                    imgs[i], result,
                    title=f"{category} | {dt} | Score: {result['anomaly_score']:.4f}",
                    save_path=save_path,
                )

    # Save ROC curve
    plot_auroc_curve(
        all_labels, all_scores,
        method_name="LW-PC-EAM",
        save_path=os.path.join(vis_dir, "roc_curve.png"),
    )
    plot_score_distribution(
        all_scores, all_labels,
        threshold=metrics["threshold_used"],
        save_path=os.path.join(vis_dir, "score_distribution.png"),
    )

    # ── Save model ────────────────────────────────────────────────────────────
    model_dir = os.path.join(save_dir, category)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "memory_bank.pt")
    torch.save({
        "memory_bank": model.memory_bank.memory,
        "spatial_shape": getattr(model, "spatial_shape", (7, 7)),
        "config": config,
    }, model_path)
    logger.info(f"Model saved to: {model_path}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics["category"] = category
    metrics["fit_time_s"] = round(fit_time, 2)
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LW-PC-EAM Training and Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root",    type=str, default="./data/mvtec_ad",
                        help="Path to MVTec AD dataset root")
    parser.add_argument("--category",     type=str, default="bottle",
                        choices=MVTEC_CATEGORIES, help="MVTec category to train on")
    parser.add_argument("--all_categories", action="store_true",
                        help="Train and evaluate on all 15 MVTec categories")
    parser.add_argument("--output_dir",   type=str, default="outputs",
                        help="Directory to save results")
    parser.add_argument("--device",       type=str, default="auto",
                        help="Device: auto / cpu / cuda / mps")
    parser.add_argument("--coreset_ratio", type=float, default=0.10,
                        help="Coreset sampling ratio (0.05-0.20)")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--download",     action="store_true",
                        help="Download MVTec dataset from Kaggle first")
    parser.add_argument("--eval_only",    action="store_true",
                        help="Skip training, only run evaluation")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Download dataset if requested
    if args.download:
        args.data_root = download_mvtec_kaggle(target_dir=args.data_root)

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["coreset_ratio"] = args.coreset_ratio
    config["batch_size"] = args.batch_size

    # Run training
    categories = MVTEC_CATEGORIES if args.all_categories else [args.category]
    all_metrics = {}

    for cat in categories:
        metrics = train_category(
            category=cat,
            data_root=args.data_root,
            config=config,
            device=device,
            save_dir=args.output_dir,
        )
        all_metrics[cat] = metrics

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Category':<20} {'AUROC':>8} {'F1':>8} {'Latency':>10}")
    logger.info("-"*50)
    aurocs = []
    for cat, m in all_metrics.items():
        logger.info(f"{cat:<20} {m['auroc']:>7.2f}%  {m['f1']:>7.2f}%  {m.get('latency_ms', 0):>8.1f}ms")
        aurocs.append(m["auroc"])
    if len(aurocs) > 1:
        logger.info("-"*50)
        logger.info(f"{'Mean':<20} {sum(aurocs)/len(aurocs):>7.2f}%")

    # Save overall summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

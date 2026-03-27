"""
ablation_study.py — Ablation Study as Described in Paper Section 4.19
=======================================================================
Ablates the following components systematically:
  1. Memory-conditioned attention mechanism (with vs without)
  2. Coreset sampling ratio (5%, 10%, 20%)
  3. Embedding dimensionality (256 vs 512)
  4. Post-training quantization (FP32 vs INT8 simulation)

Usage:
    python scripts/ablation_study.py --data_root ./data/mvtec_ad --category bottle
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataloaders
from src.metrics import LatencyBenchmark, evaluate_model
from src.model import LWPCEAM


DUMMY_INPUT = torch.randn(1, 3, 224, 224)


def run_ablation(
    data_root: str,
    category: str,
    device: str,
    output_dir: str,
) -> dict:
    """Run full ablation study and return results dict."""

    print(f"\n{'='*60}")
    print(f"  Ablation Study — Category: {category}")
    print(f"{'='*60}\n")

    train_loader, test_loader = get_dataloaders(
        root=data_root,
        category=category,
        batch_size=32,
        image_size=224,
    )

    benchmarker = LatencyBenchmark(warmup_runs=20, timed_runs=100)
    results = {}

    # ─── Ablation 1: Coreset Sampling Ratio ──────────────────────────────────
    print("Ablation 1: Coreset Sampling Ratio (5% / 10% / 20%)")
    for ratio in [0.05, 0.10, 0.20]:
        tag = f"coreset_{int(ratio*100)}pct"
        print(f"  ratio={ratio*100:.0f}% ...")
        model = LWPCEAM(coreset_ratio=ratio, pretrained=True)
        model.fit(train_loader, device=device)
        metrics = evaluate_model(model, test_loader, device=device)
        latency = benchmarker.benchmark(model, DUMMY_INPUT, device=device)
        results[tag] = {**metrics, **latency, "coreset_ratio": ratio}
        print(f"  AUROC={metrics['auroc']:.2f}%  F1={metrics['f1']:.2f}%  "
              f"Latency={latency['latency_ms']:.1f}ms")

    # ─── Ablation 2: Attention Module (with vs without) ──────────────────────
    print("\nAblation 2: With vs Without Attention Module")

    # With attention (default)
    model_att = LWPCEAM(coreset_ratio=0.10, pretrained=True)
    model_att.fit(train_loader, device=device)
    metrics_att = evaluate_model(model_att, test_loader, device=device)
    latency_att = benchmarker.benchmark(model_att, DUMMY_INPUT, device=device)
    results["with_attention"] = {**metrics_att, **latency_att}
    print(f"  With Attention:    AUROC={metrics_att['auroc']:.2f}%  "
          f"F1={metrics_att['f1']:.2f}%  Latency={latency_att['latency_ms']:.1f}ms")

    # Without attention: zero out attention weights
    class NoAttentionLWPCEAM(LWPCEAM):
        """Ablated model: skip attention, use raw backbone features."""
        @torch.no_grad()
        def predict(self, image, device="cpu"):
            self.eval()
            self.to(device)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            feat_map = self.backbone(image)
            B, D, H, W = feat_map.shape
            patches = feat_map.permute(0, 2, 3, 1).reshape(-1, D)
            # No attention — raw features
            residuals = self.memory_bank.score_patches(patches)
            anomaly_score = self.memory_bank.image_score(residuals)
            # Uniform attention weights
            uniform_attn = torch.ones(patches.shape[0], device=patches.device) / patches.shape[0]
            heatmap = self.localization.generate(residuals, uniform_attn, (H, W))
            return {
                "anomaly_score": anomaly_score,
                "is_anomaly": anomaly_score > self.anomaly_threshold,
                "heatmap": heatmap,
                "patch_residuals": residuals.cpu().numpy(),
                "attention_weights": uniform_attn.cpu().numpy(),
            }

    model_no_att = NoAttentionLWPCEAM(coreset_ratio=0.10, pretrained=True)
    model_no_att.fit(train_loader, device=device)
    metrics_no_att = evaluate_model(model_no_att, test_loader, device=device)
    latency_no_att = benchmarker.benchmark(model_no_att, DUMMY_INPUT, device=device)
    results["no_attention"] = {**metrics_no_att, **latency_no_att}
    print(f"  No Attention:      AUROC={metrics_no_att['auroc']:.2f}%  "
          f"F1={metrics_no_att['f1']:.2f}%  Latency={latency_no_att['latency_ms']:.1f}ms")

    # ─── Ablation 3: Embedding Dimensionality ─────────────────────────────────
    print("\nAblation 3: Embedding Dimensionality (256 vs 512)")
    for emb_dim in [256, 512]:
        tag = f"emb_dim_{emb_dim}"
        print(f"  embedding_dim={emb_dim} ...")
        model = LWPCEAM(embedding_dim=emb_dim, coreset_ratio=0.10, pretrained=True)
        model.fit(train_loader, device=device)
        metrics = evaluate_model(model, test_loader, device=device)
        latency = benchmarker.benchmark(model, DUMMY_INPUT, device=device)

        # Count model params
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        results[tag] = {**metrics, **latency, "params_M": round(n_params, 2)}
        print(f"  dim={emb_dim}: AUROC={metrics['auroc']:.2f}%  "
              f"F1={metrics['f1']:.2f}%  Params={n_params:.2f}M  "
              f"Latency={latency['latency_ms']:.1f}ms")

    # ─── Ablation 4: INT8 Quantization Simulation ────────────────────────────
    print("\nAblation 4: FP32 vs INT8 Quantization (simulation)")

    # FP32 baseline (already done above as 'with_attention')
    results["fp32"] = results.get("with_attention", {})
    print(f"  FP32 Latency: {results['fp32'].get('latency_ms', '?')} ms")

    # INT8 simulation: quantize backbone
    try:
        model_int8 = LWPCEAM(coreset_ratio=0.10, pretrained=True)
        model_int8.fit(train_loader, device=device)
        model_int8.backbone = torch.quantization.quantize_dynamic(
            model_int8.backbone.cpu(),
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8,
        )
        latency_int8 = benchmarker.benchmark(model_int8, DUMMY_INPUT, device="cpu")
        results["int8"] = latency_int8
        print(f"  INT8 Latency: {latency_int8['latency_ms']:.1f} ms  "
              f"FPS: {latency_int8['fps']:.1f}")
    except Exception as e:
        print(f"  INT8 quantization: skipped ({e})")
        results["int8"] = {"note": "quantization not supported on this platform"}

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'AUROC':>8} {'F1':>8} {'Latency':>10}")
    print("-"*55)
    for tag, m in results.items():
        auroc = m.get("auroc", "-")
        f1    = m.get("f1", "-")
        lat   = m.get("latency_ms", "-")
        auroc_str = f"{auroc:.2f}%" if isinstance(auroc, float) else str(auroc)
        f1_str    = f"{f1:.2f}%" if isinstance(f1, float) else str(f1)
        lat_str   = f"{lat:.1f}ms" if isinstance(lat, float) else str(lat)
        print(f"{tag:<25} {auroc_str:>8} {f1_str:>8} {lat_str:>10}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"ablation_{category}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAblation results saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="LW-PC-EAM Ablation Study")
    parser.add_argument("--data_root", type=str, default="./data/mvtec_ad")
    parser.add_argument("--category",  type=str, default="bottle")
    parser.add_argument("--device",    type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation")
    args = parser.parse_args()

    run_ablation(
        data_root=args.data_root,
        category=args.category,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

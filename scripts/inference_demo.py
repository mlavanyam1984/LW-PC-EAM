"""
inference_demo.py — Single-Image Inference Demo for LW-PC-EAM
==============================================================
Demonstrates model inference on a single image (or synthetic test image).
Generates heatmap, edge detection output, and prints anomaly score.

Usage:
    # Demo with synthetic image (no dataset needed)
    python scripts/inference_demo.py --demo

    # Real image from MVTec dataset
    python scripts/inference_demo.py --image_path ./data/mvtec_ad/bottle/test/broken_large/000.png
                                     --model_path  ./outputs/bottle/memory_bank.pt

    # Interactive batch demo on test folder
    python scripts/inference_demo.py --folder ./data/mvtec_ad/bottle/test/broken_large
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import LWPCEAM
from src.visualization import visualize_prediction, heatmap_overlay, apply_edge_detection


EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE: uses synthetic images to show pipeline without dataset
# ─────────────────────────────────────────────────────────────────────────────

def create_synthetic_normal_image() -> torch.Tensor:
    """Create a synthetic 'normal' image: uniform texture."""
    base = np.ones((224, 224, 3), dtype=np.uint8) * 180
    noise = np.random.randint(-10, 10, base.shape).astype(np.int16)
    img_arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return EVAL_TRANSFORM(Image.fromarray(img_arr))


def create_synthetic_anomaly_image() -> torch.Tensor:
    """Create a synthetic 'anomaly' image: adds a dark scratch."""
    base = np.ones((224, 224, 3), dtype=np.uint8) * 180
    noise = np.random.randint(-10, 10, base.shape).astype(np.int16)
    img_arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    # Add scratch (dark line)
    img_arr[80:140, 100:115, :] = 30
    # Add missing component (dark patch)
    img_arr[160:190, 40:80, :] = 20
    return EVAL_TRANSFORM(Image.fromarray(img_arr))


def run_demo_mode(output_dir: str = "outputs/demo") -> None:
    """
    Full demo pipeline using synthetic images.
    No dataset required.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*60)
    print("  LW-PC-EAM Inference Demo (Synthetic Images)")
    print("="*60)

    device = "cpu"
    print(f"  Device: {device}")
    print(f"  Output dir: {output_dir}\n")

    # ── Initialize model ──────────────────────────────────────────────────────
    print("[1/5] Initializing LW-PC-EAM model ...")
    model = LWPCEAM(
        embedding_dim=512,
        num_attention_heads=4,
        qk_dim=128,
        coreset_ratio=0.20,   # higher ratio for small synthetic set
        anomaly_threshold=0.3,
        gain=5.0,
        image_size=224,
        pretrained=False,     # no ImageNet weights needed for demo
    )

    # ── Build memory bank from synthetic normal images ─────────────────────────
    print("[2/5] Building memory bank from synthetic normal images ...")
    from torch.utils.data import DataLoader, TensorDataset

    normal_images = torch.stack([create_synthetic_normal_image() for _ in range(50)])
    dummy_labels  = torch.zeros(50, dtype=torch.long)
    train_ds = TensorDataset(normal_images, dummy_labels)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    model.fit(train_loader, device=device)
    print(f"  Memory bank built: {model.memory_bank.memory.shape[0]} coreset patches\n")

    # ── Inference on normal image ─────────────────────────────────────────────
    print("[3/5] Running inference on NORMAL image ...")
    normal_tensor = create_synthetic_normal_image()
    t0 = time.perf_counter()
    result_normal = model.predict(normal_tensor.unsqueeze(0), device=device)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"  Anomaly Score: {result_normal['anomaly_score']:.6f}")
    print(f"  Decision:      {'ANOMALY ⚠️' if result_normal['is_anomaly'] else 'NORMAL ✓'}")
    print(f"  Latency:       {latency_ms:.2f} ms")
    print(f"  Heatmap range: [{result_normal['heatmap'].min():.4f}, {result_normal['heatmap'].max():.4f}]")

    save_path = os.path.join(output_dir, "normal_prediction.png")
    visualize_prediction(
        normal_tensor, result_normal,
        title="Demo: Normal Image",
        save_path=save_path,
        show=False,
    )
    print(f"  Visualization saved: {save_path}\n")

    # ── Inference on anomalous image ──────────────────────────────────────────
    print("[4/5] Running inference on ANOMALOUS image (scratch + missing component) ...")
    anomaly_tensor = create_synthetic_anomaly_image()
    t0 = time.perf_counter()
    result_anomaly = model.predict(anomaly_tensor.unsqueeze(0), device=device)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"  Anomaly Score: {result_anomaly['anomaly_score']:.6f}")
    print(f"  Decision:      {'ANOMALY ⚠️' if result_anomaly['is_anomaly'] else 'NORMAL ✓'}")
    print(f"  Latency:       {latency_ms:.2f} ms")
    print(f"  Heatmap range: [{result_anomaly['heatmap'].min():.4f}, {result_anomaly['heatmap'].max():.4f}]")

    save_path = os.path.join(output_dir, "anomaly_prediction.png")
    visualize_prediction(
        anomaly_tensor, result_anomaly,
        title="Demo: Anomalous Image (Scratch + Missing Component)",
        save_path=save_path,
        show=False,
    )
    print(f"  Visualization saved: {save_path}\n")

    # ── Score comparison ──────────────────────────────────────────────────────
    print("[5/5] Score comparison summary:")
    print(f"  {'Sample':<20} {'Score':>10}  {'Decision':<15}")
    print(f"  {'-'*50}")
    print(f"  {'Normal Image':<20} {result_normal['anomaly_score']:>10.6f}  "
          f"{'NORMAL ✓' if not result_normal['is_anomaly'] else 'ANOMALY ⚠️'}")
    print(f"  {'Anomalous Image':<20} {result_anomaly['anomaly_score']:>10.6f}  "
          f"{'ANOMALY ⚠️' if result_anomaly['is_anomaly'] else 'NORMAL ✓'}")
    print()

    print("="*60)
    print(f"  Demo complete. Output files in: {output_dir}/")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# REAL IMAGE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_single_image(
    image_path: str,
    model_path: Optional[str] = None,
    output_dir: str = "outputs/inference",
    device: str = "cpu",
) -> None:
    """Run inference on a single real image."""
    import json

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    tensor = EVAL_TRANSFORM(img)

    print("Initializing model ...")
    model = LWPCEAM(pretrained=True)

    if model_path and os.path.exists(model_path):
        print(f"Loading memory bank from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.memory_bank.memory = checkpoint["memory_bank"].to(device)
        model.spatial_shape = checkpoint.get("spatial_shape", (7, 7))
    else:
        print("WARNING: No model_path provided. Running with untrained model (demo only).")

    print("Running inference ...")
    t0 = time.perf_counter()
    result = model.predict(tensor.unsqueeze(0), device=device)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\nResults:")
    print(f"  Anomaly Score: {result['anomaly_score']:.6f}")
    print(f"  Decision:      {'ANOMALY ⚠️' if result['is_anomaly'] else 'NORMAL ✓'}")
    print(f"  Latency:       {latency_ms:.2f} ms")

    fname = Path(image_path).stem
    save_path = os.path.join(output_dir, f"{fname}_prediction.png")
    visualize_prediction(tensor, result, title=fname, save_path=save_path, show=False)
    print(f"\nVisualization saved: {save_path}")

    results_dict = {
        "image": image_path,
        "anomaly_score": result["anomaly_score"],
        "is_anomaly": result["is_anomaly"],
        "latency_ms": round(latency_ms, 2),
    }
    json_path = os.path.join(output_dir, f"{fname}_result.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"JSON result saved:  {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LW-PC-EAM Inference Demo")
    parser.add_argument("--demo",       action="store_true",
                        help="Run demo with synthetic images (no dataset needed)")
    parser.add_argument("--image_path", type=str, help="Path to a single image")
    parser.add_argument("--model_path", type=str, help="Path to saved memory bank .pt")
    parser.add_argument("--output_dir", type=str, default="outputs/demo")
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    if args.demo or args.image_path is None:
        run_demo_mode(output_dir=args.output_dir)
    else:
        run_single_image(
            image_path=args.image_path,
            model_path=args.model_path,
            output_dir=args.output_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()

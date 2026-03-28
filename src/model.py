"""
LW-PC-EAM: Lightweight PatchCore with Explainable Attention Mechanism
=======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
import torchvision.models as models
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LIGHTWEIGHT CNN BACKBONE (MobileNetV2)
# ─────────────────────────────────────────────────────────────────────────────

class LightweightBackbone(nn.Module):
    """
    MobileNetV2-based feature extractor.
    Extracts multi-scale patch features from intermediate layers.
    Input: RGB image (B, 3, 224, 224)
    Output: patch feature tensor (B, C, H, W)
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        base = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)

        # Use layers up to layer 14 (feature map ~14x14 at 224x224 input)
        self.features = base.features[:14]

        # Project to desired embedding dimension
        in_channels = 96  # MobileNetV2 layer-14 output channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.embedding_dim = embedding_dim

        # Freeze backbone weights (edge-deployment setting)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized input image
        Returns:
            features: (B, embedding_dim, H', W') patch feature map
        """
        feat = self.features(x)          # (B, 96, H', W')
        feat = self.proj(feat)           # (B, 512, H', W')
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# 2. MEMORY-CONDITIONED ATTENTION MODULE
# ─────────────────────────────────────────────────────────────────────────────

class MemoryConditionedAttention(nn.Module):
    """
    Explainable attention mechanism conditioned on coreset residuals.
    - 4 attention heads
    - 128-dimensional query/key projections
    - Produces attention coefficients b_q per patch (Eq. 3)
    - Attention weights are DIRECTLY used for explainability maps
      (no post-hoc gradient computation required)

    Forward pass residual-weighted integration:
        v_q = X_at * (b_q ⊙ g_q)    [Equation 3 from paper]
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 4,
        qk_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.embedding_dim = embedding_dim

        # Query and Key projections (128-dim as per paper)
        self.q_proj = nn.Linear(embedding_dim, qk_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(embedding_dim, qk_dim * num_heads, bias=False)

        # Output projection (attention-weighted features → embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.scale = qk_dim ** -0.5

    def forward(
        self,
        patch_features: torch.Tensor,          # (N, embedding_dim)
        coreset_residuals: Optional[torch.Tensor] = None,  # (N,) residual scores
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_features: (N, D) patch embeddings from backbone
            coreset_residuals: (N,) optional residual scores for conditioning
        Returns:
            attended_features: (N, D) attention-weighted patch embeddings
            attention_weights: (N,) per-patch attention coefficient (for heatmap)
        """
        N, D = patch_features.shape

        # Compute Q, K
        Q = self.q_proj(patch_features).view(N, self.num_heads, self.qk_dim)  # (N, H, qk)
        K = self.k_proj(patch_features).view(N, self.num_heads, self.qk_dim)  # (N, H, qk)

        # Scaled dot-product attention across patches
        # (H, N, qk) x (H, qk, N) → (H, N, N)
        Q_t = Q.permute(1, 0, 2)
        K_t = K.permute(1, 0, 2)
        attn_logits = torch.bmm(Q_t, K_t.transpose(1, 2)) * self.scale  # (H, N, N)
        attn_map = F.softmax(attn_logits, dim=-1)                         # (H, N, N)

        # Mean over heads → per-patch attention coefficient b_q
        b_q = attn_map.mean(dim=0).mean(dim=-1)  # (N,)

        # Condition on coreset residuals if provided
        if coreset_residuals is not None:
            residual_weight = torch.sigmoid(coreset_residuals)
            b_q = b_q * residual_weight

        # Normalize attention weights
        b_q = b_q / (b_q.sum() + 1e-8)

        # Attention-weighted projection: v_q = X_at * (b_q ⊙ g_q)  [Eq. 3]
        weighted = patch_features * b_q.unsqueeze(-1)        # (N, D) element-wise
        attended = self.out_proj(weighted)                    # (N, D)

        return attended, b_q


# ─────────────────────────────────────────────────────────────────────────────
# 3. PATCHCORE MEMORY BANK WITH CORESET SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

class PatchCoreMemoryBank:
    """
    PatchCore-style memory bank with greedy minimax coreset sampling.

    Training:
      - Collect all normal-sample patch embeddings
      - Subsample to coreset_ratio (default 10%) via greedy minimax facility location
      - Store coreset as the memory bank

    Inference:
      - Compute Euclidean distance from query patch to nearest coreset exemplar
      - τ_q = ||v_q - ∂||₂   [Equation 4 from paper]
      - Image score = max patch score
    """

    def __init__(self, coreset_ratio: float = 0.10):
        self.coreset_ratio = coreset_ratio
        self.memory: Optional[torch.Tensor] = None  # (M, D)

    def build(self, features: torch.Tensor) -> None:
        """
        Build coreset memory bank from normal training features.
        Args:
            features: (N, D) all patch embeddings from normal training images
        """
        logger.info(f"Building memory bank from {features.shape[0]} patches ...")
        coreset = self._greedy_minimax_coreset(features)
        self.memory = coreset
        logger.info(f"Memory bank size: {self.memory.shape[0]} patches "
                    f"({self.coreset_ratio*100:.0f}% of {features.shape[0]})")

    def _greedy_minimax_coreset(self, features: torch.Tensor) -> torch.Tensor:
        """
        Greedy approximation of minimax facility location problem.
        Iteratively selects the point farthest from the current selected set.
        """
        n = features.shape[0]
        target_size = max(1, int(n * self.coreset_ratio))

        # Work on CPU for memory efficiency
        feats = features.cpu().float()

        selected_idx = [torch.randint(0, n, (1,)).item()]
        min_dists = torch.full((n,), float("inf"))

        for _ in range(target_size - 1):
            last = feats[selected_idx[-1]].unsqueeze(0)  # (1, D)
            dists = torch.cdist(feats, last).squeeze(1)   # (N,)
            min_dists = torch.minimum(min_dists, dists)
            next_idx = min_dists.argmax().item()
            selected_idx.append(next_idx)

        selected = torch.stack([feats[i] for i in selected_idx])
        return selected.to(features.device)

    def score_patches(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly residuals for query patches.
        τ_q = ||v_q - ∂||₂   [Equation 4]
        Args:
            query_features: (N, D)
        Returns:
            residuals: (N,) Euclidean distances to nearest exemplar
        """
        assert self.memory is not None, "Memory bank not built. Call .build() first."
        # (N, M) pairwise distances
        dists = torch.cdist(
            query_features.float(),
            self.memory.float()
        )
        residuals, _ = dists.min(dim=1)  # (N,) nearest-neighbor distance
        return residuals

    def image_score(self, patch_residuals: torch.Tensor) -> float:
        """Image-level anomaly score = max patch residual."""
        return patch_residuals.max().item()


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPLAINABLE LOCALIZATION MAP GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ExplainableLocalizationMap:
    """
    Generates interpretable anomaly heatmaps using logistic squashing.

    Δ(y,z) = ∂_q(y,z) · ρ(l · (σ_q - δ))    [Equation 2]

    where:
        ∂_q(y,z)  = indicator (patch membership)
        ρ          = logistic / sigmoid function
        l          = gain (sharpness)
        δ          = operational threshold
        σ_q        = patch-wise residual [Equation 1]
    """

    def __init__(
        self,
        gain: float = 5.0,
        threshold: float = 0.5,
        output_size: Tuple[int, int] = (224, 224),
    ):
        self.gain = gain
        self.threshold = threshold
        self.output_size = output_size

    def generate(
        self,
        patch_residuals: torch.Tensor,  # (N,) — σ_q values
        attention_weights: torch.Tensor,  # (N,) — b_q from attention
        spatial_shape: Tuple[int, int],   # (H', W') feature map dims
    ) -> np.ndarray:
        """
        Args:
            patch_residuals:  (N,) raw Euclidean residuals per patch
            attention_weights: (N,) attention coefficients
            spatial_shape:    (H', W') grid dimensions
        Returns:
            heatmap: (H, W) normalized [0,1] localization map
        """
        H, W = spatial_shape
        N = H * W
        assert patch_residuals.shape[0] == N, \
            f"Expected {N} patches, got {patch_residuals.shape[0]}"

        # Equation 1: patch-wise residual σ_q (already computed)
        sigma_q = patch_residuals.float()

        # Equation 2: logistic squashing
        activation = torch.sigmoid(self.gain * (sigma_q - self.threshold))

        # Weight by attention coefficients (memory-conditioned)
        weighted_activation = activation * (attention_weights + 1e-8)

        # Reshape to spatial grid
        heatmap = weighted_activation.view(H, W).cpu().numpy()

        # Upsample to original image resolution
        heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        heatmap_up = F.interpolate(
            heatmap_tensor,
            size=self.output_size,
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        # Normalize to [0, 1]
        mn, mx = heatmap_up.min(), heatmap_up.max()
        if mx - mn > 1e-8:
            heatmap_up = (heatmap_up - mn) / (mx - mn)

        return heatmap_up


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL LW-PC-EAM FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

class LWPCEAM(nn.Module):
    """
    LW-PC-EAM: Lightweight PatchCore with Explainable Attention Mechanism

    Full pipeline (Algorithm 2 from paper):
      1. Preprocess image (normalize, resize)
      2. Extract patch features via MobileNetV2 backbone
      3. Apply memory-conditioned attention (4 heads, 128-dim Q/K)
      4. Score patches via PatchCore memory bank (coreset Euclidean distance)
      5. Generate explainable heatmap (logistic squashing, no post-hoc gradients)
      6. Return: anomaly score + heatmap + edge detection output
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_attention_heads: int = 4,
        qk_dim: int = 128,
        coreset_ratio: float = 0.10,
        anomaly_threshold: float = 0.5,
        gain: float = 5.0,
        image_size: int = 224,
        pretrained: bool = True,
    ):
        super().__init__()

        self.backbone = LightweightBackbone(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
        )
        self.attention = MemoryConditionedAttention(
            embedding_dim=embedding_dim,
            num_heads=num_attention_heads,
            qk_dim=qk_dim,
        )
        self.memory_bank = PatchCoreMemoryBank(coreset_ratio=coreset_ratio)
        self.localization = ExplainableLocalizationMap(
            gain=gain,
            threshold=anomaly_threshold,
            output_size=(image_size, image_size),
        )

        self.anomaly_threshold = anomaly_threshold
        self.image_size = image_size

    # ------------------------------------------------------------------
    # Training: build memory bank from normal images
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, dataloader: torch.utils.data.DataLoader, device: str = "cpu") -> None:
        """
        Build PatchCore memory bank from normal (defect-free) training images.
        Args:
            dataloader: yields (images, labels) — only normal images used
            device:     'cpu', 'cuda', or 'mps'
        """
        self.eval()
        self.to(device)
        if IPEX_AVAILABLE and device == "cpu":
            self.backbone = ipex.optimize(self.backbone)
            self.attention = ipex.optimize(self.attention)
        all_features: List[torch.Tensor] = []

        for batch in dataloader:
            imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            feat_map = self.backbone(imgs)           # (B, D, H', W')
            B, D, H, W = feat_map.shape
            patches = feat_map.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
            all_features.append(patches.cpu())

        all_features_t = torch.cat(all_features, dim=0)
        self.memory_bank.build(all_features_t.to(device))
        self.spatial_shape = (H, W)
        logger.info(f"Training complete. Spatial grid: {H}x{W}")

    # ------------------------------------------------------------------
    # Inference: detect anomalies + generate heatmap
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,   # (1, 3, H, W) or (3, H, W)
        device: str = "cpu",
    ) -> dict:
        """
        Run full LW-PC-EAM inference pipeline on a single image.

        Returns dict with:
            'anomaly_score':   float — image-level anomaly score
            'is_anomaly':      bool  — True if score > threshold
            'heatmap':         np.ndarray (H, W) — localization map [0,1]
            'patch_residuals': np.ndarray (N,)   — per-patch scores
            'attention_weights': np.ndarray (N,) — per-patch attention
        """
        self.eval()
        self.to(device)

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        # Step 1: Extract patch features
        feat_map = self.backbone(image)                  # (1, D, H', W')
        B, D, H, W = feat_map.shape
        patches = feat_map.permute(0, 2, 3, 1).reshape(-1, D)  # (H'*W', D)

        # Step 2: Initial residuals (for attention conditioning)
        raw_residuals = self.memory_bank.score_patches(patches)  # (N,)

        # Step 3: Memory-conditioned attention
        attended_patches, attention_weights = self.attention(
            patches, coreset_residuals=raw_residuals
        )

        # Step 4: Final residual scoring on attended features  [Eq. 4]
        final_residuals = self.memory_bank.score_patches(attended_patches)  # (N,)

        # Step 5: Image-level score
        anomaly_score = self.memory_bank.image_score(final_residuals)

        # Step 6: Explainable heatmap  [Eq. 2]
        heatmap = self.localization.generate(
            patch_residuals=final_residuals,
            attention_weights=attention_weights,
            spatial_shape=(H, W),
        )

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score > self.anomaly_threshold,
            "heatmap": heatmap,
            "patch_residuals": final_residuals.cpu().numpy(),
            "attention_weights": attention_weights.cpu().numpy(),
        }

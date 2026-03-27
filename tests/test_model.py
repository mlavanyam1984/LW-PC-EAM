"""
tests/test_model.py
====================
Unit tests for LW-PC-EAM components.
Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Try importing torch; skip GPU tests gracefully if unavailable ─────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")

from src.model import (
    LightweightBackbone,
    MemoryConditionedAttention,
    PatchCoreMemoryBank,
    ExplainableLocalizationMap,
    LWPCEAM,
)
from src.metrics import (
    compute_auroc,
    compute_precision_recall_f1,
    compute_detection_accuracy,
    localization_fidelity,
    clarity_index,
    operational_cost,
    reconstruction_error,
    similarity_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    return torch.randn(1, 3, 224, 224)

@pytest.fixture
def dummy_batch():
    return torch.randn(4, 3, 224, 224)

@pytest.fixture
def patch_features():
    return torch.randn(196, 512)   # 14×14 spatial grid, 512-dim embeddings

@pytest.fixture
def small_memory():
    """Small memory bank for fast tests."""
    bank = PatchCoreMemoryBank(coreset_ratio=0.5)
    bank.memory = torch.randn(20, 512)
    return bank


# ─────────────────────────────────────────────────────────────────────────────
# 1. BACKBONE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestLightweightBackbone:
    def test_output_shape(self, dummy_image):
        backbone = LightweightBackbone(embedding_dim=512, pretrained=False)
        out = backbone(dummy_image)
        assert out.dim() == 4, "Output should be 4D (B, C, H, W)"
        assert out.shape[1] == 512, "Channel dim should equal embedding_dim"

    def test_batch_output_shape(self, dummy_batch):
        backbone = LightweightBackbone(embedding_dim=512, pretrained=False)
        out = backbone(dummy_batch)
        assert out.shape[0] == 4, "Batch size should be preserved"
        assert out.shape[1] == 512

    def test_backbone_frozen(self):
        backbone = LightweightBackbone(pretrained=False)
        for param in backbone.features.parameters():
            assert not param.requires_grad, "Backbone features should be frozen"

    def test_proj_trainable(self):
        backbone = LightweightBackbone(pretrained=False)
        for param in backbone.proj.parameters():
            assert param.requires_grad, "Projection head should be trainable"

    def test_embedding_dim_256(self, dummy_image):
        backbone = LightweightBackbone(embedding_dim=256, pretrained=False)
        out = backbone(dummy_image)
        assert out.shape[1] == 256


# ─────────────────────────────────────────────────────────────────────────────
# 2. ATTENTION MODULE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryConditionedAttention:
    def test_output_shapes(self, patch_features):
        attn = MemoryConditionedAttention(embedding_dim=512, num_heads=4, qk_dim=128)
        attended, weights = attn(patch_features)
        assert attended.shape == patch_features.shape, "attended shape should match input"
        assert weights.shape == (196,), "weights should be (N,)"

    def test_attention_weights_sum_to_one(self, patch_features):
        attn = MemoryConditionedAttention(embedding_dim=512, num_heads=4, qk_dim=128)
        _, weights = attn(patch_features)
        assert abs(weights.sum().item() - 1.0) < 1e-4, "Attention weights should sum to 1"

    def test_attention_weights_nonnegative(self, patch_features):
        attn = MemoryConditionedAttention(embedding_dim=512, num_heads=4, qk_dim=128)
        _, weights = attn(patch_features)
        assert (weights >= 0).all(), "Attention weights should be non-negative"

    def test_with_residual_conditioning(self, patch_features):
        attn = MemoryConditionedAttention(embedding_dim=512, num_heads=4, qk_dim=128)
        residuals = torch.rand(196)
        attended, weights = attn(patch_features, coreset_residuals=residuals)
        assert attended.shape == patch_features.shape

    def test_num_heads_2(self, patch_features):
        attn = MemoryConditionedAttention(embedding_dim=512, num_heads=2, qk_dim=64)
        attended, weights = attn(patch_features)
        assert attended.shape == patch_features.shape


# ─────────────────────────────────────────────────────────────────────────────
# 3. MEMORY BANK TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchCoreMemoryBank:
    def test_build_coreset_size(self):
        bank = PatchCoreMemoryBank(coreset_ratio=0.10)
        features = torch.randn(1000, 512)
        bank.build(features)
        expected = max(1, int(1000 * 0.10))
        assert bank.memory.shape[0] == expected, f"Expected {expected} coreset patches"

    def test_coreset_5pct(self):
        bank = PatchCoreMemoryBank(coreset_ratio=0.05)
        features = torch.randn(500, 512)
        bank.build(features)
        assert bank.memory.shape[0] == 25

    def test_coreset_20pct(self):
        bank = PatchCoreMemoryBank(coreset_ratio=0.20)
        features = torch.randn(200, 512)
        bank.build(features)
        assert bank.memory.shape[0] == 40

    def test_score_patches_shape(self, small_memory, patch_features):
        residuals = small_memory.score_patches(patch_features)
        assert residuals.shape == (196,), "Residuals should be (N,)"

    def test_score_patches_nonnegative(self, small_memory, patch_features):
        residuals = small_memory.score_patches(patch_features)
        assert (residuals >= 0).all(), "Euclidean distances should be non-negative"

    def test_image_score_is_max(self, small_memory):
        residuals = torch.tensor([0.1, 0.8, 0.3, 0.5])
        score = small_memory.image_score(residuals)
        assert abs(score - 0.8) < 1e-5, "Image score should be max patch residual"

    def test_not_built_raises(self, patch_features):
        bank = PatchCoreMemoryBank()
        with pytest.raises(AssertionError):
            bank.score_patches(patch_features)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOCALIZATION MAP TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainableLocalizationMap:
    def test_output_shape(self):
        mapper = ExplainableLocalizationMap(output_size=(224, 224))
        residuals = torch.rand(196)
        weights   = torch.rand(196)
        heatmap   = mapper.generate(residuals, weights, spatial_shape=(14, 14))
        assert heatmap.shape == (224, 224), "Heatmap should be (224, 224)"

    def test_output_range(self):
        mapper = ExplainableLocalizationMap(output_size=(224, 224))
        residuals = torch.rand(196) * 2
        weights   = torch.ones(196) / 196
        heatmap   = mapper.generate(residuals, weights, spatial_shape=(14, 14))
        assert heatmap.min() >= 0.0, "Heatmap should be >= 0"
        assert heatmap.max() <= 1.0, "Heatmap should be <= 1"

    def test_high_residuals_give_high_activation(self):
        mapper = ExplainableLocalizationMap(gain=10.0, threshold=0.3, output_size=(14, 14))
        residuals_high = torch.ones(196) * 2.0
        residuals_low  = torch.zeros(196)
        weights = torch.ones(196) / 196
        hm_high = mapper.generate(residuals_high, weights, (14, 14))
        hm_low  = mapper.generate(residuals_low,  weights, (14, 14))
        assert hm_high.mean() > hm_low.mean(), "High residuals → high activation"

    def test_wrong_patch_count_raises(self):
        mapper = ExplainableLocalizationMap()
        with pytest.raises(AssertionError):
            mapper.generate(torch.rand(100), torch.rand(100), spatial_shape=(14, 14))


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestLWPCEAMPipeline:
    @pytest.fixture
    def trained_model(self):
        """Build a minimal trained model using synthetic data."""
        from torch.utils.data import DataLoader, TensorDataset
        model = LWPCEAM(
            embedding_dim=512, num_attention_heads=4, qk_dim=128,
            coreset_ratio=0.30, anomaly_threshold=0.5,
            image_size=224, pretrained=False,
        )
        # 8 synthetic normal training images
        imgs = torch.randn(8, 3, 224, 224)
        labels = torch.zeros(8, dtype=torch.long)
        loader = DataLoader(TensorDataset(imgs, labels), batch_size=4)
        model.fit(loader, device="cpu")
        return model

    def test_predict_returns_required_keys(self, trained_model, dummy_image):
        result = trained_model.predict(dummy_image, device="cpu")
        for key in ["anomaly_score", "is_anomaly", "heatmap",
                    "patch_residuals", "attention_weights"]:
            assert key in result, f"Missing key: {key}"

    def test_predict_heatmap_shape(self, trained_model, dummy_image):
        result = trained_model.predict(dummy_image, device="cpu")
        assert result["heatmap"].shape == (224, 224)

    def test_predict_score_is_float(self, trained_model, dummy_image):
        result = trained_model.predict(dummy_image, device="cpu")
        assert isinstance(result["anomaly_score"], float)

    def test_predict_is_anomaly_is_bool(self, trained_model, dummy_image):
        result = trained_model.predict(dummy_image, device="cpu")
        assert isinstance(result["is_anomaly"], bool)

    def test_predict_3d_input(self, trained_model):
        img = torch.randn(3, 224, 224)   # no batch dim
        result = trained_model.predict(img, device="cpu")
        assert result["heatmap"].shape == (224, 224)

    def test_memory_bank_built_after_fit(self, trained_model):
        assert trained_model.memory_bank.memory is not None
        assert trained_model.memory_bank.memory.shape[1] == 512


# ─────────────────────────────────────────────────────────────────────────────
# 6. METRICS TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_auroc_perfect(self):
        labels = [0]*50 + [1]*50
        scores = list(np.linspace(0.0, 0.49, 50)) + list(np.linspace(0.51, 1.0, 50))
        auroc, thresh = compute_auroc(labels, scores)
        assert auroc > 0.99, "Perfect separation should give AUROC > 0.99"

    def test_auroc_random(self):
        np.random.seed(0)
        labels = [0, 1] * 50
        scores = np.random.rand(100).tolist()
        auroc, _ = compute_auroc(labels, scores)
        assert 0.0 <= auroc <= 1.0

    def test_precision_recall_f1(self):
        labels = [0, 0, 1, 1, 1, 0]
        preds  = [0, 1, 1, 1, 0, 0]
        result = compute_precision_recall_f1(labels, preds)
        assert "precision" in result and "recall" in result and "f1" in result
        assert 0 <= result["f1"] <= 100

    def test_detection_accuracy_all_correct(self):
        labels = [0, 0, 1, 1]
        preds  = [0, 0, 1, 1]
        acc = compute_detection_accuracy(labels, preds)
        assert abs(acc - 100.0) < 1e-4

    def test_localization_fidelity_range(self):
        heatmaps = [np.random.rand(224, 224) for _ in range(5)]
        gts = [(np.random.rand(224, 224) > 0.8).astype(float) for _ in range(5)]
        lf = localization_fidelity(heatmaps, gts)
        assert 0.0 <= lf <= 1.0

    def test_clarity_index_range(self):
        heatmaps = [np.random.rand(224, 224) for _ in range(5)]
        ci = clarity_index(heatmaps)
        assert 0.0 <= ci <= 1.0

    def test_operational_cost_weights_sum_to_one(self):
        # Default weights β=0.45, γ=0.30, δ=0.25 sum to 1.0
        assert abs(0.45 + 0.30 + 0.25 - 1.0) < 1e-9

    def test_operational_cost_formula(self):
        cost = operational_cost(100, 50, 30, beta=0.45, gamma=0.30, delta=0.25)
        expected = 0.45*100 + 0.30*50 + 0.25*30
        assert abs(cost - expected) < 1e-6

    def test_reconstruction_error_zero_for_identical(self):
        y = np.random.rand(100)
        err = reconstruction_error(y, y)
        assert abs(err) < 1e-10

    def test_similarity_score_identical_vectors(self):
        v = np.random.rand(512)
        score = similarity_score(v, v)
        assert abs(score - 1.0) < 1e-5

    def test_similarity_score_range(self):
        v1 = np.random.randn(512)
        v2 = np.random.randn(512)
        score = similarity_score(v1, v2)
        assert -1.0 <= score <= 1.0

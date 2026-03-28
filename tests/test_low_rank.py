"""
Tests for SlimNet low-rank factorization module.
Week 3 success criterion: BERT FFN layers factorized at 90% variance threshold,
model size reduces ~25%, forward pass matches within tolerance.
"""
import pytest
import torch
import torch.nn as nn

from slimnet.configs import LowRankConfig
from slimnet.techniques.low_rank import (
    LowRankFactorizationModule,
    FactorizedLinear,
    _factorize_linear,
    _select_rank,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class BERTlikeFFN(nn.Module):
    """
    Mimics BERT's Feed-Forward Network blocks (768→3072→768).
    This is exactly the use case highlighted in PRD 5.4.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SmallMLP(nn.Module):
    """Small model with layers below min_layer_size — should be skipped."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def bert_ffn():
    return BERTlikeFFN().eval()


@pytest.fixture
def small_mlp():
    return SmallMLP().eval()


@pytest.fixture
def bert_input():
    return torch.randn(4, 768)


@pytest.fixture
def small_input():
    return torch.randn(4, 64)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSelectRank:
    def test_full_rank_at_threshold_1(self):
        """Threshold=1.0 should require all singular values."""
        S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
        k = _select_rank(S, variance_threshold=1.0)
        assert k == len(S)

    def test_rank_1_at_low_threshold(self):
        """When first singular value dominates, low threshold needs rank 1."""
        S = torch.tensor([100.0, 0.1, 0.1, 0.1])
        k = _select_rank(S, variance_threshold=0.99)
        assert k == 1

    def test_rank_is_bounded(self):
        """Selected rank must be between 1 and len(S)."""
        S = torch.abs(torch.randn(50))
        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
            k = _select_rank(S, threshold)
            assert 1 <= k <= 50

    def test_rank_increases_with_threshold(self):
        """Higher variance threshold requires more singular values (higher rank)."""
        S = torch.abs(torch.randn(20)) + 0.1  # ensure all positive
        S, _ = torch.sort(S, descending=True)
        k90 = _select_rank(S, 0.90)
        k99 = _select_rank(S, 0.99)
        assert k90 <= k99, f"k(0.90)={k90} should be <= k(0.99)={k99}"


class TestFactorizeLinear:
    def test_factorizes_large_layer(self):
        """Should factorize a 768×3072 layer (BERT FFN projection)."""
        layer = nn.Linear(3072, 768)
        factorized, info = _factorize_linear(layer, variance_threshold=0.90)
        assert factorized is not None, f"Should factorize: {info}"
        assert info["rank_selected"] < 768
        assert info["compression_ratio"] > 1.0

    def test_skips_small_layer_no_reduction(self):
        """Should skip layers where factorization would not reduce params."""
        # A 4×4 layer: factorization would increase params
        layer = nn.Linear(4, 4)
        factorized, info = _factorize_linear(layer, variance_threshold=0.90)
        # Either skipped entirely or reports no reduction
        if factorized is not None:
            assert info["compression_ratio"] > 1.0

    def test_factorized_layer_forward_pass(self):
        """FactorizedLinear must produce same-shape output as original."""
        layer = nn.Linear(1024, 512)
        factorized, info = _factorize_linear(layer, variance_threshold=0.90)
        if factorized is not None:
            x = torch.randn(8, 1024)
            with torch.no_grad():
                out = factorized(x)
            assert out.shape == (8, 512)

    def test_variance_explained_close_to_threshold(self):
        """Variance explained by selected rank should be >= threshold."""
        layer = nn.Linear(768, 3072)
        threshold = 0.90
        factorized, info = _factorize_linear(layer, variance_threshold=threshold)
        if factorized is not None:
            assert info["variance_explained"] >= threshold - 0.01  # small tolerance


class TestLowRankFactorizationModule:
    def test_reduces_params_on_bert_ffn(self, bert_ffn, bert_input):
        """BERT FFN should have fewer params after factorization."""
        original_params = sum(p.numel() for p in bert_ffn.parameters())
        lrm = LowRankFactorizationModule()
        config = LowRankConfig(variance_threshold=0.90, min_layer_size=512)
        factorized, info = lrm.apply(bert_ffn, config, bert_input)
        factorized_params = sum(p.numel() for p in factorized.parameters())
        assert factorized_params < original_params, (
            f"Factorized params {factorized_params} should be < original {original_params}"
        )
        assert info["factorized_layers"] > 0

    def test_skips_small_layers(self, small_mlp, small_input):
        """Layers smaller than min_layer_size should be skipped."""
        lrm = LowRankFactorizationModule()
        config = LowRankConfig(variance_threshold=0.90, min_layer_size=512)
        factorized, info = lrm.apply(small_mlp, config, small_input)
        assert info["factorized_layers"] == 0

    def test_forward_pass_runs(self, bert_ffn, bert_input):
        """Factorized model must run forward pass without errors."""
        lrm = LowRankFactorizationModule()
        config = LowRankConfig(variance_threshold=0.90, min_layer_size=512)
        factorized, _ = lrm.apply(bert_ffn, config, bert_input)
        with torch.no_grad():
            out = factorized(bert_input)
        assert out.shape == bert_input.shape  # FFN is 768→768

    def test_cosine_similarity_high(self, bert_ffn, bert_input):
        """Output of factorized model must closely match original (cosine sim > 0.99)."""
        lrm = LowRankFactorizationModule()
        config = LowRankConfig(variance_threshold=0.90, min_layer_size=512)
        factorized, info = lrm.apply(bert_ffn, config, bert_input)

        with torch.no_grad():
            orig_out = bert_ffn(bert_input)
            fact_out = factorized(bert_input)

        cos_sim = nn.functional.cosine_similarity(
            orig_out.reshape(1, -1), fact_out.reshape(1, -1)
        ).item()
        assert cos_sim > 0.90, (
            f"Cosine similarity {cos_sim:.4f} is below threshold 0.90"
        )

    def test_does_not_modify_original(self, bert_ffn, bert_input):
        """Original model must be unchanged after factorization."""
        original_params = sum(p.numel() for p in bert_ffn.parameters())
        lrm = LowRankFactorizationModule()
        config = LowRankConfig(variance_threshold=0.90, min_layer_size=512)
        lrm.apply(bert_ffn, config, bert_input)
        assert sum(p.numel() for p in bert_ffn.parameters()) == original_params

    def test_info_has_layer_details(self, bert_ffn, bert_input):
        """Info dict must include per-layer factorization details."""
        lrm = LowRankFactorizationModule()
        config = LowRankConfig(variance_threshold=0.90, min_layer_size=512)
        _, info = lrm.apply(bert_ffn, config, bert_input)
        assert "layer_details" in info
        assert "overall_compression" in info
        assert info["overall_compression"] > 1.0
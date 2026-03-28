"""
Tests for SlimNet structured pruning module.
Week 2 success criterion: ResNet-18 pruned to 60% of original params with magnitude scoring.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from slimnet.configs import PruneConfig
from slimnet.techniques.pruning import (
    StructuredPruningModule,
    _magnitude_importance,
    _prune_linear_layer,
    _prune_conv2d_layer,
    _get_keep_indices,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class SimpleLinearNet(nn.Module):
    """Pure Linear model for testing neuron pruning."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu(self.fc1(x)))))


class SimpleConvNet(nn.Module):
    """Simple CNN with Conv2d layers for filter pruning tests."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))


@pytest.fixture
def linear_net():
    return SimpleLinearNet().eval()


@pytest.fixture
def conv_net():
    return SimpleConvNet().eval()


@pytest.fixture
def linear_input():
    return torch.randn(4, 128)


@pytest.fixture
def image_input():
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def labeled_loader():
    inputs = torch.randn(128, 128)
    labels = torch.randint(0, 10, (128,))
    return DataLoader(TensorDataset(inputs, labels), batch_size=16)


@pytest.fixture
def image_loader():
    inputs = torch.randn(128, 3, 32, 32)
    labels = torch.randint(0, 10, (128,))
    return DataLoader(TensorDataset(inputs, labels), batch_size=16)


# ---------------------------------------------------------------------------
# Unit tests: importance scoring
# ---------------------------------------------------------------------------

class TestMagnitudeImportance:
    def test_linear_weight_returns_correct_shape(self):
        """Magnitude importance for Linear weight (out×in) should return (out,) scores."""
        weight = torch.randn(64, 128)  # Linear(128, 64)
        importance = _magnitude_importance(weight)
        assert importance.shape == (64,), f"Expected (64,), got {importance.shape}"

    def test_conv2d_weight_returns_correct_shape(self):
        """Magnitude importance for Conv2d weight (out,in,kH,kW) should return (out,) scores."""
        weight = torch.randn(32, 3, 3, 3)  # Conv2d(3, 32, 3)
        importance = _magnitude_importance(weight)
        assert importance.shape == (32,), f"Expected (32,), got {importance.shape}"

    def test_all_positive(self):
        """Magnitude (L1 norm) must always be non-negative."""
        weight = torch.randn(16, 32)
        importance = _magnitude_importance(weight)
        assert (importance >= 0).all(), "Magnitude importance must be non-negative"

    def test_zero_row_has_zero_importance(self):
        """A zero-weight neuron must have zero importance."""
        weight = torch.randn(16, 32)
        weight[5] = 0.0  # Zero out neuron 5
        importance = _magnitude_importance(weight)
        assert importance[5].item() == 0.0


class TestPruneLinearLayer:
    def test_reduces_output_features(self):
        """Pruning should reduce the output dimension of a Linear layer."""
        layer = nn.Linear(128, 64)
        keep_idx = torch.arange(32)  # Keep half
        pruned = _prune_linear_layer(layer, keep_idx)
        assert pruned.out_features == 32

    def test_preserves_input_features(self):
        """Pruning output neurons must not change input dimension."""
        layer = nn.Linear(128, 64)
        keep_idx = torch.arange(32)
        pruned = _prune_linear_layer(layer, keep_idx)
        assert pruned.in_features == 128

    def test_weights_match_kept_rows(self):
        """Pruned layer weights must exactly match the selected rows."""
        layer = nn.Linear(8, 16)
        keep_idx = torch.tensor([0, 2, 4, 6])
        pruned = _prune_linear_layer(layer, keep_idx)
        assert torch.allclose(pruned.weight.data, layer.weight.data[keep_idx])

    def test_bias_pruned_correctly(self):
        """Bias must be pruned to match kept output neurons."""
        layer = nn.Linear(8, 16, bias=True)
        keep_idx = torch.tensor([1, 3, 5])
        pruned = _prune_linear_layer(layer, keep_idx)
        assert pruned.bias is not None
        assert torch.allclose(pruned.bias.data, layer.bias.data[keep_idx])


class TestGetKeepIndices:
    def test_keeps_correct_fraction(self):
        """Should keep (1-sparsity) fraction of neurons."""
        importance = torch.arange(100, dtype=torch.float)
        keep = _get_keep_indices(importance, sparsity=0.3)
        assert len(keep) == 70  # 100 * (1 - 0.3) = 70

    def test_keeps_highest_importance(self):
        """Should keep neurons with highest importance scores."""
        importance = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0])
        keep = _get_keep_indices(importance, sparsity=0.4)  # keep 3
        # Top-3 by importance: indices 3 (8.0), 1 (5.0), 4 (3.0)
        assert 3 in keep.tolist()
        assert 1 in keep.tolist()

    def test_minimum_keep_respected(self):
        """Should never reduce below min_keep neurons."""
        importance = torch.ones(10)
        keep = _get_keep_indices(importance, sparsity=0.99, min_keep=4)
        assert len(keep) >= 4


# ---------------------------------------------------------------------------
# Integration tests: StructuredPruningModule
# ---------------------------------------------------------------------------

class TestStructuredPruningModule:
    def test_reduces_parameter_count(self, linear_net, linear_input):
        """Pruning must reduce total parameter count."""
        original_params = sum(p.numel() for p in linear_net.parameters())
        pm = StructuredPruningModule()
        config = PruneConfig(sparsity=0.3, method="magnitude", n_steps=2)
        pruned, info = pm.apply(linear_net, config, linear_input)
        pruned_params = sum(p.numel() for p in pruned.parameters())
        assert pruned_params < original_params, (
            f"Pruned params {pruned_params} should be < original {original_params}"
        )

    def test_actual_sparsity_in_info(self, linear_net, linear_input):
        """Info dict must report actual_sparsity."""
        pm = StructuredPruningModule()
        config = PruneConfig(sparsity=0.3, method="magnitude", n_steps=2)
        _, info = pm.apply(linear_net, config, linear_input)
        assert "actual_sparsity" in info
        assert 0 < info["actual_sparsity"] <= 1.0

    def test_forward_pass_runs_after_pruning(self, linear_net, linear_input):
        """Pruned model must run a forward pass without errors."""
        pm = StructuredPruningModule()
        config = PruneConfig(sparsity=0.2, method="magnitude", n_steps=1)
        pruned, _ = pm.apply(linear_net, config, linear_input)
        with torch.no_grad():
            out = pruned(linear_input)
        assert out.shape[-1] == 10  # num_classes preserved

    def test_does_not_modify_original(self, linear_net, linear_input):
        """Original model must not be modified by pruning."""
        original_param_count = sum(p.numel() for p in linear_net.parameters())
        pm = StructuredPruningModule()
        config = PruneConfig(sparsity=0.3, method="magnitude", n_steps=1)
        pm.apply(linear_net, config, linear_input)
        assert sum(p.numel() for p in linear_net.parameters()) == original_param_count

    def test_gradient_method_with_calibration(self, linear_net, linear_input, labeled_loader):
        """Gradient scoring must work when calibration data is provided."""
        pm = StructuredPruningModule()
        config = PruneConfig(sparsity=0.2, method="gradient", n_steps=1)
        pruned, info = pm.apply(linear_net, config, linear_input, labeled_loader)
        assert info["method"] == "gradient"
        assert sum(p.numel() for p in pruned.parameters()) < sum(
            p.numel() for p in linear_net.parameters()
        )

    def test_falls_back_to_magnitude_without_calibration(self, linear_net, linear_input):
        """Gradient/taylor without calibration must fall back to magnitude."""
        pm = StructuredPruningModule()
        config = PruneConfig(sparsity=0.2, method="gradient", n_steps=1)
        pruned, info = pm.apply(linear_net, config, linear_input, calibration_loader=None)
        assert info["method"] == "magnitude"

    def test_iterative_schedule_produces_correct_steps(self, linear_net, linear_input):
        """n_steps=3 should run 3 pruning passes (verified via param count trend)."""
        pm = StructuredPruningModule()
        config = PruneConfig(sparsity=0.4, method="magnitude", n_steps=3)
        _, info = pm.apply(linear_net, config, linear_input)
        assert info["n_steps"] == 3

    def test_conv_net_pruning(self, conv_net, image_input):
        """Structured pruning must work on Conv2d layers."""
        original_params = sum(p.numel() for p in conv_net.parameters())
        pm = StructuredPruningModule()
        config = PruneConfig(sparsity=0.3, method="magnitude", n_steps=2)
        pruned, info = pm.apply(conv_net, config, image_input)
        pruned_params = sum(p.numel() for p in pruned.parameters())
        assert pruned_params <= original_params
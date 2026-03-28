"""
Low-Rank Factorization Module — PRD §5.4
SVD decomposition: W ≈ U × S × Vᵀ, keep top-k singular values.
Linear(m,n) → Linear(m,k) → Linear(k,n)
Applied to Linear layers with min(m,n) >= min_layer_size.
"""
from __future__ import annotations
import copy, logging
from typing import Optional
import torch
import torch.nn as nn
from slimnet.configs import LowRankConfig

logger = logging.getLogger(__name__)


class FactorizedLinear(nn.Module):
    """Replaces a single nn.Linear with two smaller ones after SVD."""
    def __init__(self, in_f: int, out_f: int, rank: int, bias: Optional[torch.Tensor]):
        super().__init__()
        self.layer_A = nn.Linear(in_f, rank, bias=False)    # (rank, in_f)
        self.layer_B = nn.Linear(rank, out_f, bias=bias is not None)
        if bias is not None:
            self.layer_B.bias = nn.Parameter(bias.clone())
        self.rank = rank
        self.original_in = in_f
        self.original_out = out_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_B(self.layer_A(x))

    def extra_repr(self) -> str:
        return f"in={self.original_in}, out={self.original_out}, rank={self.rank}"


class LowRankFactorizationModule:
    """
    Usage:
        lrm = LowRankFactorizationModule()
        compressed, info = lrm.apply(model, config, dummy_input)
    """

    def apply(
        self,
        model: nn.Module,
        config: LowRankConfig,
        dummy_input: Optional[torch.Tensor] = None,
    ) -> tuple[nn.Module, dict]:
        model = copy.deepcopy(model)
        orig_params = sum(p.numel() for p in model.parameters())

        self._replace(model, config)

        new_params = sum(p.numel() for p in model.parameters())
        n_factorized = sum(1 for m in model.modules() if isinstance(m, FactorizedLinear))
        reduction = (1 - new_params / orig_params) * 100 if orig_params > 0 else 0

        logger.info(
            f"[LowRank] factorized {n_factorized} layers. "
            f"Params: {orig_params:,} → {new_params:,} ({reduction:.1f}% reduction)"
        )
        info = {
            "n_factorized": n_factorized,
            "original_params": orig_params,
            "new_params": new_params,
            "param_reduction_pct": reduction,
        }
        return model, info

    def _replace(self, module: nn.Module, config: LowRankConfig, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                factorized = self._factorize(child, full, config)
                if factorized is not None:
                    setattr(module, name, factorized)
            else:
                self._replace(child, config, full)

    def _factorize(self, layer: nn.Linear, name: str, config: LowRankConfig) -> Optional[FactorizedLinear]:
        m, n = layer.out_features, layer.in_features
        if min(m, n) < config.min_layer_size:
            return None

        W = layer.weight.data.float()
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except torch.linalg.LinAlgError as e:
            logger.warning(f"[LowRank] SVD failed on {name}: {e}")
            return None

        # PRD §5.4.1: keep top-k singular values explaining variance_threshold of variance
        S_sq = S ** 2
        cumvar = torch.cumsum(S_sq, dim=0) / S_sq.sum()
        k = int(torch.searchsorted(cumvar, config.variance_threshold).item()) + 1
        k = max(1, min(k, S.shape[0]))

        # Skip if factorization increases param count
        if k * (m + n) >= m * n:
            logger.debug(f"[LowRank] skip {name}: rank {k} gives no reduction")
            return None

        device = W.device
        fl = FactorizedLinear(n, m, k, layer.bias)
        with torch.no_grad():
            # layer_A weight: (k, n) = diag(S[:k]) @ Vt[:k]
            fl.layer_A.weight.data = (S[:k].unsqueeze(1) * Vt[:k]).to(device)
            # layer_B weight: (m, k) = U[:, :k]
            fl.layer_B.weight.data = U[:, :k].to(device)

        reduction = (1 - k * (m + n) / (m * n)) * 100
        logger.info(f"[LowRank] {name}: ({m},{n}) → rank {k} ({reduction:.1f}% reduction)")
        return fl
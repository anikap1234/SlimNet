"""
SlimNet configuration dataclasses — PRD §5
All technique configs plus hardware target map live here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Hardware target → VRAM budget (MB).  Also used by planner + pipeline.
# ---------------------------------------------------------------------------
TARGET_HARDWARE_PRESETS: dict[str, dict] = {
    "t4":      {"vram_budget_mb": 15360.0, "cpu_only": False},
    "cpu":     {"vram_budget_mb": 0.0,     "cpu_only": True},
    "macbook": {"vram_budget_mb": 16384.0, "cpu_only": False},
    "rtx3070": {"vram_budget_mb": 7680.0,  "cpu_only": False},
    "rtx3080": {"vram_budget_mb": 10240.0, "cpu_only": False},
    "rtx3060": {"vram_budget_mb": 5120.0,  "cpu_only": False},
}


@dataclass
class QuantConfig:
    """PRD §5.2 — quantization configuration."""
    mode: Literal["dynamic", "static", "fp16", "bf16"] = "dynamic"
    per_channel: bool = True
    validation_threshold: float = 0.999


@dataclass
class PruneConfig:
    """PRD §5.3 — structured pruning configuration."""
    sparsity: float = 0.3
    method: Literal["magnitude", "gradient", "taylor"] = "magnitude"
    n_steps: int = 5
    min_layer_params: int = 1024


@dataclass
class LowRankConfig:
    """PRD §5.4 — SVD low-rank factorization configuration."""
    variance_threshold: float = 0.90
    min_layer_size: int = 512


@dataclass
class DistillConfig:
    """PRD §5.5 — knowledge distillation configuration."""
    temperature: float = 4.0
    alpha: float = 0.3        # weight on hard-label task loss
    beta: float = 0.7         # weight on KL distillation loss
    epochs: int = 3
    intermediate: bool = False
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    grad_clip: float = 1.0


@dataclass
class CompressionStep:
    """One step inside a CompressionPlan."""
    technique: Literal["quantization", "structured_pruning", "low_rank", "distillation"]
    config: QuantConfig | PruneConfig | LowRankConfig | DistillConfig
    reason: str = ""          # human-readable why this step was chosen


@dataclass
class CompressionPlan:
    """Ordered list of compression steps. Distillation always last."""
    steps: list[CompressionStep] = field(default_factory=list)
    estimated_size_mb: float = 0.0
    estimated_latency_ms: float = 0.0
    estimated_accuracy_drop: float = 0.0

    def __repr__(self) -> str:
        lines = ["CompressionPlan("]
        for s in self.steps:
            lines.append(f"  {s.technique}")
        lines.append(f"  est_size={self.estimated_size_mb:.1f} MB")
        lines.append(f"  est_latency={self.estimated_latency_ms:.1f} ms")
        lines.append(")")
        return "\n".join(lines)
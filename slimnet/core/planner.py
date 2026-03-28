"""
HardwareProfiler and CompressionPlanner — PRD §5.1
"""
from __future__ import annotations
import time, platform, dataclasses
from typing import Optional
import psutil
import torch
import torch.nn as nn
from slimnet.configs import (
    TARGET_HARDWARE_PRESETS,
    CompressionPlan, CompressionStep,
    QuantConfig, PruneConfig, LowRankConfig, DistillConfig,
)


@dataclasses.dataclass
class HardwareProfile:
    """PRD §5.1.1"""
    vram_total_mb: float
    vram_free_mb: float
    ram_total_mb: float
    device_type: str           # 'cuda' | 'mps' | 'cpu'
    cpu_arch: str
    baseline_latency_ms: float
    baseline_memory_mb: float

    def __repr__(self) -> str:
        return (
            f"HardwareProfile(device={self.device_type}, "
            f"vram_free={self.vram_free_mb:.0f} MB, "
            f"ram={self.ram_total_mb:.0f} MB, "
            f"latency={self.baseline_latency_ms:.2f} ms, "
            f"mem={self.baseline_memory_mb:.1f} MB)"
        )


class HardwareProfiler:
    def __init__(self) -> None:
        self._device = self._detect_device()

    def profile(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        n_warmup: int = 5,
        n_timed: int = 20,
    ) -> HardwareProfile:
        vram_total, vram_free = self._get_vram()
        ram_total = psutil.virtual_memory().total / (1024 ** 2)
        cpu_arch = self._get_cpu_arch()

        model = model.to(self._device).eval()
        inp = sample_input.to(self._device)

        with torch.no_grad():
            for _ in range(n_warmup):
                model(inp)

        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

        latencies: list[float] = []
        with torch.no_grad():
            for _ in range(n_timed):
                t0 = time.perf_counter()
                model(inp)
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                latencies.append((time.perf_counter() - t0) * 1000)

        mean_latency = sum(latencies) / len(latencies)

        if self._device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)
        else:
            peak_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 ** 2)

        return HardwareProfile(
            vram_total_mb=vram_total,
            vram_free_mb=vram_free,
            ram_total_mb=ram_total,
            device_type=self._device.type,
            cpu_arch=cpu_arch,
            baseline_latency_ms=mean_latency,
            baseline_memory_mb=peak_mb,
        )

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _get_vram(self) -> tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024 ** 2)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        return total, total - allocated

    def _get_cpu_arch(self) -> str:
        m = platform.machine().lower()
        if "arm" in m or "aarch" in m:
            return "Apple Silicon" if platform.system() == "Darwin" else "arm64"
        return "x86_64"


class CompressionPlanner:
    """
    Rule-based planner (v1). Produces a CompressionPlan from hardware profile
    and user constraints. PRD §5.1.2 heuristics implemented verbatim.
    """

    def plan(
        self,
        profile: HardwareProfile,
        model: nn.Module,
        target: str,
        max_accuracy_drop: float = 0.02,
    ) -> CompressionPlan:
        preset = TARGET_HARDWARE_PRESETS.get(target.lower(), {"vram_budget_mb": 0.0, "cpu_only": True})
        target_vram = preset["vram_budget_mb"]
        is_cpu_target = preset["cpu_only"] or target_vram == 0.0
        current_memory_mb = profile.baseline_memory_mb

        steps: list[CompressionStep] = []

        # Rule 1: Quantization
        quant_needed = is_cpu_target or target_vram < 0.25 * current_memory_mb or max_accuracy_drop >= 0.03
        if quant_needed:
            mode = "dynamic" if is_cpu_target else "static"
            steps.append(CompressionStep(
                technique="quantization",
                config=QuantConfig(mode=mode),
                reason=f"CPU target={is_cpu_target}, mode={mode}",
            ))

        # Rule 2: Structured Pruning
        # PRD: max_accuracy_drop < 1% → skip pruning
        if max_accuracy_drop >= 0.01:
            if max_accuracy_drop > 0.05:
                sparsity = 0.50
            elif max_accuracy_drop >= 0.03:
                sparsity = 0.40
            else:
                sparsity = 0.30
            steps.append(CompressionStep(
                technique="structured_pruning",
                config=PruneConfig(sparsity=sparsity),
                reason=f"sparsity={sparsity}",
            ))

        # Rule 3: Low-rank factorization only for large models
        if current_memory_mb > 200.0 and max_accuracy_drop >= 0.01:
            steps.append(CompressionStep(
                technique="low_rank",
                config=LowRankConfig(),
                reason="model > 200MB",
            ))

        # Rule 4: Distillation — ALWAYS last
        epochs = 3 if max_accuracy_drop >= 0.02 else 5
        steps.append(CompressionStep(
            technique="distillation",
            config=DistillConfig(epochs=epochs),
            reason="accuracy recovery — always last",
        ))

        raw_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        est_ratio = self._estimate_ratio(steps)
        return CompressionPlan(
            steps=steps,
            estimated_size_mb=raw_size / est_ratio,
            estimated_latency_ms=profile.baseline_latency_ms / min(est_ratio, 4.0),
            estimated_accuracy_drop=max_accuracy_drop * 0.5,
        )

    def _estimate_ratio(self, steps: list[CompressionStep]) -> float:
        ratio = 1.0
        for step in steps:
            if step.technique == "quantization":
                ratio *= 3.5 if step.config.mode in ("dynamic", "static") else 1.8  # type: ignore
            elif step.technique == "structured_pruning":
                ratio *= 1.0 / (1.0 - step.config.sparsity)  # type: ignore
            elif step.technique == "low_rank":
                ratio *= 1.3
        return ratio
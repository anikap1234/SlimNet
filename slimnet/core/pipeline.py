"""
Compression Pipeline and Compressor — PRD §5 / Week 5

IMPORTANT ordering fix:
    PRD says quantization first, but dynamic int8 quantization replaces
    nn.Linear with DynamicQuantizedLinear whose parameters() is empty —
    breaking pruning and distillation optimizers downstream.

    Correct order for CPU dynamic quant:
        structured_pruning → distillation → quantization

    For static quant / fp16 / bf16 (parameters still accessible):
        quantization → structured_pruning → distillation
"""
from __future__ import annotations
import copy, logging
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slimnet.configs import (
    TARGET_HARDWARE_PRESETS,
    CompressionPlan, CompressionStep,
    QuantConfig, PruneConfig, LowRankConfig, DistillConfig,
)
from slimnet.core.planner import HardwareProfiler, CompressionPlanner
from slimnet.core.benchmark import BenchmarkReport, BenchmarkReporter
from slimnet.techniques.quantization import QuantizationModule
from slimnet.techniques.pruning import StructuredPruningModule
from slimnet.techniques.low_rank import LowRankFactorizationModule
from slimnet.techniques.distillation import KnowledgeDistillationModule

logger = logging.getLogger(__name__)


class CompressedModel:
    """
    Return value of slimnet.compress() and Compressor.fit() — PRD §6.1.

    Attributes:
        model:   Compressed nn.Module, ready for inference.
        report:  BenchmarkReport with size, latency, accuracy stats.
        plan:    CompressionPlan that was executed.
    """
    def __init__(self, model: nn.Module, report: BenchmarkReport,
                 plan: CompressionPlan, technique_infos: dict):
        self.model = model
        self.report = report
        self.plan = plan
        self.technique_infos = technique_infos

    def save(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "report": {
                "original_size_mb":   self.report.original_size_mb,
                "compressed_size_mb": self.report.compressed_size_mb,
                "compression_ratio":  self.report.compression_ratio,
                "speedup":            self.report.speedup,
                "accuracy_delta":     self.report.accuracy_delta,
                "techniques_applied": self.report.techniques_applied,
                "target":             self.report.target,
            },
        }, path)
        logger.info(f"Saved to {path}")

    def __repr__(self) -> str:
        return (
            f"CompressedModel(size={self.report.compressed_size_mb:.2f} MB, "
            f"ratio={self.report.compression_ratio:.2f}x, "
            f"speedup={self.report.speedup:.2f}x, "
            f"techniques={self.report.techniques_applied})"
        )


class CompressionPipeline:
    """
    Executes a CompressionPlan step by step.

    Key design decision: dynamic int8 quantization is applied LAST on CPU
    because it converts nn.Linear → DynamicQuantizedLinear, which has no
    trainable parameters — breaking any optimizer that runs after it.
    The planner marks quantization steps with needs_last=True for this case.
    """

    def __init__(self):
        self._quantizer  = QuantizationModule()
        self._pruner     = StructuredPruningModule()
        self._factorizer = LowRankFactorizationModule()
        self._distiller  = KnowledgeDistillationModule()

    def execute(
        self,
        model: nn.Module,
        plan: CompressionPlan,
        dummy_input: torch.Tensor,
        calibration_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        max_accuracy_drop: float,
        original_accuracy: Optional[float],
    ) -> tuple[nn.Module, dict, list[str]]:
        """
        Execute plan. Returns (compressed_model, per_technique_info, applied_list).
        Original model is never modified — deep copy made internally.

        Reorders steps so that dynamic quantization always runs after
        pruning and distillation (avoids empty parameter list error).
        """
        steps = self._reorder_steps(plan.steps)

        current = copy.deepcopy(model)
        teacher = copy.deepcopy(model)   # frozen copy kept for distillation
        tech_infos: dict = {}
        applied: list[str] = []
        device = self._device()

        for step in steps:
            t   = step.technique
            cfg = step.config
            logger.info(f"[Pipeline] → {t.upper()}  ({step.reason})")

            try:
                if t == "quantization":
                    current, info = self._quantizer.apply(
                        current, cfg, dummy_input, calibration_loader
                    )
                    tech_infos["quantization"] = info
                    applied.append("quantization")

                elif t == "structured_pruning":
                    current, info = self._pruner.apply(
                        current, cfg, dummy_input, calibration_loader
                    )
                    tech_infos["pruning"] = info
                    applied.append("pruning")

                elif t == "low_rank":
                    current, info = self._factorizer.apply(
                        current, cfg, dummy_input
                    )
                    tech_infos["low_rank"] = info
                    applied.append("low_rank")

                elif t == "distillation":
                    if calibration_loader is None:
                        logger.warning("[Pipeline] skipping distillation — no calibration_loader")
                        continue
                    current, info = self._distiller.train(
                        teacher=copy.deepcopy(teacher),
                        student=current,
                        train_loader=calibration_loader,
                        config=cfg,
                        val_loader=val_loader,
                        max_accuracy_drop=max_accuracy_drop,
                        original_accuracy=original_accuracy,
                        device=device,
                    )
                    tech_infos["distillation"] = info
                    applied.append("distillation")

                else:
                    logger.warning(f"[Pipeline] unknown technique '{t}' — skipping")

            except Exception as e:
                import traceback
                logger.error(f"[Pipeline] step '{t}' failed: {e}\n{traceback.format_exc()}")
                logger.warning(f"[Pipeline] continuing without {t}")

        return current, tech_infos, applied

    @staticmethod
    def _reorder_steps(steps: list[CompressionStep]) -> list[CompressionStep]:
        """
        Move dynamic/static quantization to AFTER pruning and distillation.

        Dynamic int8 quant replaces nn.Linear with DynamicQuantizedLinear —
        model.parameters() returns empty after that, so any subsequent
        optimizer call raises ValueError("optimizer got an empty parameter list").

        Safe ordering:
            structured_pruning → low_rank → distillation → quantization

        fp16 / bf16 casts keep parameters accessible, so they can stay first
        if desired — but moving them last is always safe.
        """
        quant_steps    = [s for s in steps if s.technique == "quantization"]
        non_quant      = [s for s in steps if s.technique != "quantization"]

        # distillation must be last among non-quant steps (PRD requirement)
        distill_steps  = [s for s in non_quant if s.technique == "distillation"]
        other_steps    = [s for s in non_quant if s.technique != "distillation"]

        reordered = other_steps + distill_steps + quant_steps

        if quant_steps:
            logger.info(
                "[Pipeline] reordered: quantization moved after pruning/distillation "
                "to preserve optimizer parameter access"
            )
        return reordered

    @staticmethod
    def _device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class Compressor:
    """
    Advanced API — PRD §6.2.

    Usage:
        compressor = Compressor(
            quantization=QuantConfig(mode='dynamic'),
            pruning=PruneConfig(sparsity=0.4, method='taylor', n_steps=5),
            low_rank=LowRankConfig(variance_threshold=0.90, min_layer_size=512),
            distillation=DistillConfig(temperature=4.0, alpha=0.3, beta=0.7, epochs=5),
        )
        result = compressor.fit(model, calibration_data, val_data=val_loader)
    """

    def __init__(
        self,
        quantization: Optional[QuantConfig] = None,
        pruning: Optional[PruneConfig] = None,
        low_rank: Optional[LowRankConfig] = None,
        distillation: Optional[DistillConfig] = None,
    ):
        steps = []
        if quantization is not None:
            steps.append(CompressionStep("quantization", quantization, "user-specified"))
        if pruning is not None:
            steps.append(CompressionStep("structured_pruning", pruning, "user-specified"))
        if low_rank is not None:
            steps.append(CompressionStep("low_rank", low_rank, "user-specified"))
        dist_cfg = distillation if distillation is not None else DistillConfig()
        steps.append(CompressionStep("distillation", dist_cfg, "accuracy recovery — always last"))

        self._plan     = CompressionPlan(steps=steps)
        self._pipeline = CompressionPipeline()
        self._reporter = BenchmarkReporter()
        self._distiller = KnowledgeDistillationModule()

    def fit(
        self,
        model: nn.Module,
        calibration_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        dummy_input: Optional[torch.Tensor] = None,
        max_accuracy_drop: float = 0.02,
        target: str = "cpu",
    ) -> CompressedModel:
        if dummy_input is None:
            dummy_input = _first_input(calibration_data)

        device = CompressionPipeline._device()
        original_accuracy = None
        if val_data is not None:
            original_accuracy = self._distiller._evaluate(
                model.to(device).eval(), val_data, device
            )
            logger.info(f"[Compressor] original accuracy: {original_accuracy:.4f}")

        compressed, tech_infos, applied = self._pipeline.execute(
            model=model, plan=self._plan, dummy_input=dummy_input,
            calibration_loader=calibration_data, val_loader=val_data,
            max_accuracy_drop=max_accuracy_drop, original_accuracy=original_accuracy,
        )

        preset = TARGET_HARDWARE_PRESETS.get(target, {"vram_budget_mb": 0.0})
        report = self._reporter.compare(
            original_model=model, compressed_model=compressed,
            dummy_input=dummy_input, target=target,
            target_vram_budget_mb=preset["vram_budget_mb"],
            val_loader=val_data, techniques_applied=applied,
        )
        return CompressedModel(
            model=compressed, report=report,
            plan=self._plan, technique_infos=tech_infos
        )


def _first_input(loader: DataLoader) -> torch.Tensor:
    batch = next(iter(loader))
    inp = batch[0] if isinstance(batch, (list, tuple)) else batch
    return inp[:1]
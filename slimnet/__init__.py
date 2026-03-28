"""
SlimNet — PyTorch model compression in one function call.  PRD §6.1
"""
from __future__ import annotations
import logging
from typing import Optional

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public re-exports
from slimnet.configs import (
    QuantConfig, PruneConfig, LowRankConfig, DistillConfig,
    CompressionPlan, CompressionStep, TARGET_HARDWARE_PRESETS,
)
from slimnet.core.pipeline import CompressedModel, Compressor, CompressionPipeline, _first_input
from slimnet.core.benchmark import BenchmarkReport, BenchmarkReporter
from slimnet.core.planner import HardwareProfiler, CompressionPlanner, HardwareProfile

__version__ = "0.1.0"
__all__ = [
    "compress",
    "Compressor",
    "CompressedModel",
    "QuantConfig", "PruneConfig", "LowRankConfig", "DistillConfig",
    "BenchmarkReport",
    "__version__",
]


def compress(
    model,
    calibration_data,
    target: str = "cpu",
    max_accuracy_drop: float = 0.02,
    val_data=None,
    sample_input=None,
    verbose: bool = True,
) -> CompressedModel:
    """
    Compress any PyTorch model to fit a target hardware constraint.

    PRD §6.1 — Simple API targeting 90% of users.

    Args:
        model:              Any nn.Module.
        calibration_data:   DataLoader, min 512 samples. Labels not required.
        target:             't4' | 'cpu' | 'macbook' | 'rtx3070' | 'rtx3080'
        max_accuracy_drop:  Max accuracy degradation allowed (0.02 = 2%).
        val_data:           Optional DataLoader — enables accuracy reporting.
        sample_input:       Optional sample tensor. Auto-inferred if None.
        verbose:            Print benchmark report to stdout.

    Returns:
        CompressedModel with .model, .report, .save()

    Example:
        import slimnet
        result = slimnet.compress(model, loader, target='t4')
        print(result.report)
        result.model.eval()
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="[slimnet] %(message)s")

    if target not in TARGET_HARDWARE_PRESETS:
        raise ValueError(
            f"Unknown target '{target}'. "
            f"Valid: {list(TARGET_HARDWARE_PRESETS.keys())}"
        )

    # Auto-infer sample_input
    if sample_input is None:
        if calibration_data is None:
            raise ValueError(
                "Provide sample_input= or calibration_data= so SlimNet can "
                "infer the input shape for hardware profiling."
            )
        sample_input = _first_input(calibration_data)

    # 1. Profile hardware
    profiler = HardwareProfiler()
    hw_profile = profiler.profile(model, sample_input)
    logger.info(f"Hardware: {hw_profile}")

    # 2. Build compression plan
    planner = CompressionPlanner()
    plan = planner.plan(
        profile=hw_profile,
        model=model,
        target=target,
        max_accuracy_drop=max_accuracy_drop,
    )
    logger.info(f"Plan: {[s.technique for s in plan.steps]}")

    # 3. Measure original accuracy (optional)
    original_accuracy: Optional[float] = None
    if val_data is not None:
        from slimnet.techniques.distillation import KnowledgeDistillationModule
        device = CompressionPipeline._device()
        original_accuracy = KnowledgeDistillationModule()._evaluate(
            model.to(device).eval(), val_data, device
        )
        logger.info(f"Original accuracy: {original_accuracy:.4f}")

    # 4. Execute compression pipeline
    pipeline = CompressionPipeline()
    compressed, tech_infos, applied = pipeline.execute(
        model=model,
        plan=plan,
        dummy_input=sample_input,
        calibration_loader=calibration_data,
        val_loader=val_data,
        max_accuracy_drop=max_accuracy_drop,
        original_accuracy=original_accuracy,
    )

    # 5. Build benchmark report
    preset = TARGET_HARDWARE_PRESETS[target]
    reporter = BenchmarkReporter()
    report = reporter.compare(
        original_model=model,
        compressed_model=compressed,
        dummy_input=sample_input,
        target=target,
        target_vram_budget_mb=preset["vram_budget_mb"],
        val_loader=val_data,
        techniques_applied=applied,
    )

    if verbose:
        print(str(report))

    return CompressedModel(model=compressed, report=report, plan=plan, technique_infos=tech_infos)


logger = logging.getLogger(__name__)
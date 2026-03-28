"""
SlimNet — PyTorch Model Compression Library
============================================

Compress any PyTorch model to run on cheap hardware.
One function call. No ML expertise required.

Basic usage:
    import slimnet

    compressed = slimnet.compress(
        model=my_model,
        calibration_data=dataloader,
        target='t4',
        max_accuracy_drop=0.02,
    )
    compressed.model      # nn.Module, ready for inference
    compressed.report     # BenchmarkReport
    compressed.save('model.slimnet')

Advanced usage:
    from slimnet import Compressor, QuantConfig, PruneConfig, LowRankConfig, DistillConfig

    compressor = Compressor(
        quantization=QuantConfig(mode='int8', per_channel=True),
        pruning=PruneConfig(sparsity=0.4, method='taylor', n_steps=5),
        low_rank=LowRankConfig(variance_threshold=0.90, min_layer_size=512),
        distillation=DistillConfig(temperature=4.0, alpha=0.3, beta=0.7, epochs=5),
    )
    result = compressor.fit(model, calibration_data, val_data=val_loader)
"""

import logging

# Set up library-level logging (users configure their own handlers)
logging.getLogger(__name__).addHandler(logging.NullHandler())

from slimnet.configs import (
    QuantConfig,
    PruneConfig,
    LowRankConfig,
    DistillConfig,
    TARGET_HARDWARE_PRESETS,
)
from slimnet.core.pipeline import Compressor, CompressedModel
from slimnet.core.benchmark import BenchmarkReport
from slimnet.core.planner import HardwareProfiler, CompressionPlanner, HardwareProfile, CompressionPlan

__version__ = "0.1.0"
__author__ = "SlimNet Contributors"
__all__ = [
    "compress",
    "Compressor",
    "CompressedModel",
    "QuantConfig",
    "PruneConfig",
    "LowRankConfig",
    "DistillConfig",
    "BenchmarkReport",
    "HardwareProfiler",
    "CompressionPlanner",
    "HardwareProfile",
    "CompressionPlan",
    "TARGET_HARDWARE_PRESETS",
]


def compress(
    model,
    calibration_data,
    target: str = "cpu",
    max_accuracy_drop: float = 0.02,
    val_data=None,
    dummy_input=None,
    verbose: bool = True,
) -> CompressedModel:
    """
    Compress a PyTorch model to run on a target hardware constraint.

    This is the simple one-function API (PRD 6.1) targeting 90% of users.
    SlimNet automatically selects and applies the best combination of:
      - Dynamic/static int8 quantization
      - Structured pruning (neurons, filters, attention heads)
      - SVD-based low-rank factorization
      - Knowledge distillation for accuracy recovery

    Args:
        model: Any torch.nn.Module.
        calibration_data: torch.utils.data.DataLoader with >= 512 samples.
                          Labels are NOT required for distillation — only
                          teacher forward passes are used.
        target: Hardware target preset string.
                One of: 't4' | 'cpu' | 'macbook' | 'rtx3070' | 'rtx3080'
        max_accuracy_drop: Maximum tolerable accuracy degradation (0.02 = 2%).
                           Lower values → more conservative compression.
        val_data: Optional DataLoader for accuracy evaluation.
                  If provided, BenchmarkReport includes accuracy_delta.
        dummy_input: Optional representative input tensor.
                     If None, auto-inferred from first batch of calibration_data.
        verbose: If True, enable INFO-level logging to stdout.

    Returns:
        CompressedModel with:
            .model   — compressed nn.Module, ready for inference
            .report  — BenchmarkReport (size, latency, accuracy, etc.)
            .plan    — CompressionPlan that was executed
            .save(path) — save to disk

    Example:
        import slimnet
        import torchvision.models as models

        model = models.resnet50(pretrained=True)
        compressed = slimnet.compress(model, calibration_loader, target='t4')
        print(compressed.report)
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="[slimnet] %(message)s",
        )

    if target not in TARGET_HARDWARE_PRESETS:
        raise ValueError(
            f"Unknown target '{target}'. "
            f"Valid targets: {list(TARGET_HARDWARE_PRESETS.keys())}"
        )

    # Auto-infer dummy_input from first batch
    if dummy_input is None:
        batch = next(iter(calibration_data))
        if isinstance(batch, (list, tuple)):
            dummy_input = batch[0][:1]
        else:
            dummy_input = batch[:1]

    # Profile hardware
    profiler = HardwareProfiler()
    profile = profiler.profile(model, dummy_input)

    logging.getLogger(__name__).info(
        f"Hardware: {profile.device_type.upper()} "
        f"({profile.cuda_device_name}) | "
        f"Model: {profile.model_size_mb:.1f} MB | "
        f"Baseline latency: {profile.baseline_latency_ms:.2f} ms"
    )

    # Build compression plan
    planner = CompressionPlanner()
    plan = planner.plan(
        profile=profile,
        target=target,
        max_accuracy_drop=max_accuracy_drop,
        has_calibration_data=True,
    )

    logging.getLogger(__name__).info(
        f"Compression plan: {[s.technique for s in plan.steps]}"
    )
    logging.getLogger(__name__).info(
        f"Estimated: {plan.estimated_size_mb:.1f} MB, "
        f"{plan.estimated_latency_ms:.1f} ms, "
        f"~{plan.estimated_accuracy_drop:.1%} accuracy drop"
    )

    # Execute pipeline
    from slimnet.core.pipeline import CompressionPipeline
    from slimnet.core.benchmark import BenchmarkReporter

    pipeline = CompressionPipeline()
    reporter = BenchmarkReporter()

    # Measure original accuracy if val_data provided
    original_accuracy = None
    if val_data is not None:
        logging.getLogger(__name__).info("Measuring original model accuracy...")
        device = pipeline._get_training_device()
        from slimnet.techniques.distillation import KnowledgeDistillationModule
        original_accuracy = KnowledgeDistillationModule._evaluate(
            model.to(device).eval(), val_data, device
        )
        logging.getLogger(__name__).info(f"Original accuracy: {original_accuracy:.4f}")

    compressed, tech_infos, techniques = pipeline.execute(
        model=model,
        plan=plan,
        dummy_input=dummy_input,
        calibration_loader=calibration_data,
        val_loader=val_data,
        max_accuracy_drop=max_accuracy_drop,
        original_accuracy=original_accuracy,
    )

    preset = TARGET_HARDWARE_PRESETS[target]
    report = reporter.compare(
        original_model=model,
        compressed_model=compressed,
        dummy_input=dummy_input,
        target=target,
        target_vram_budget_mb=preset["vram_budget_mb"],
        val_loader=val_data,
        techniques_applied=techniques,
    )

    if verbose:
        print(str(report))

    return CompressedModel(
        model=compressed,
        report=report,
        plan=plan,
        technique_infos=tech_infos,
    )
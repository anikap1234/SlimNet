"""
SlimNet CLI  (PRD Week 5)

Usage:
    slimnet compress model.pt --target t4 --max-drop 0.02
    slimnet profile model.pt
    slimnet info
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(
    name="slimnet",
    help="SlimNet — PyTorch model compression in one command.",
    add_completion=False,
)
console = Console()


@app.command()
def compress(
    model_path: Path = typer.Argument(..., help="Path to a TorchScript or state_dict .pt file"),
    target: str = typer.Option("cpu", "--target", "-t", help="Hardware target: t4 | cpu | macbook | rtx3070"),
    max_drop: float = typer.Option(0.02, "--max-drop", "-d", help="Max accuracy drop (default 0.02 = 2%)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for compressed model"),
    calibration: Optional[Path] = typer.Option(None, "--calibration", "-c", help="Path to calibration data tensor (.pt)"),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """
    Compress a PyTorch model to a target hardware constraint.

    Example:
        slimnet compress resnet50.pt --target t4 --max-drop 0.02
    """
    import torch

    if verbose:
        logging.basicConfig(level=logging.INFO, format="[slimnet] %(message)s")

    console.print(Panel.fit(
        f"[bold cyan]SlimNet[/] — Compressing [yellow]{model_path.name}[/] → [green]{target}[/]",
        border_style="cyan",
    ))

    # Load model
    with console.status("Loading model..."):
        try:
            model = torch.load(str(model_path), map_location="cpu")
            if not hasattr(model, "forward"):
                console.print("[red]Error:[/] Loaded object is not an nn.Module. "
                              "Save with torch.save(model, path) not state_dict.")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Failed to load model:[/] {e}")
            raise typer.Exit(1)

    # Load or create dummy calibration data
    if calibration is not None:
        with console.status("Loading calibration data..."):
            calib_tensor = torch.load(str(calibration), map_location="cpu")
    else:
        console.print("[yellow]No calibration data provided — using random dummy data.[/]")
        # Try to infer input shape from model
        try:
            first_param = next(model.parameters())
            dummy = torch.randn(16, 3, 224, 224)  # Default: ImageNet-like
        except StopIteration:
            dummy = torch.randn(16, 3, 224, 224)
        calib_tensor = dummy

    # Wrap tensor in a DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    if isinstance(calib_tensor, torch.Tensor):
        dataset = TensorDataset(calib_tensor)
    else:
        dataset = calib_tensor

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Run compression
    import slimnet

    with console.status(f"[cyan]Compressing (target={target}, max_drop={max_drop:.1%})...[/]"):
        result = slimnet.compress(
            model=model,
            calibration_data=loader,
            target=target,
            max_accuracy_drop=max_drop,
            verbose=verbose,
        )

    # Print report table
    _print_report_table(result.report)

    # Save output
    out_path = output or model_path.with_suffix(".slimnet.pt")
    result.save(str(out_path))
    console.print(f"\n[green]✓[/] Compressed model saved to [bold]{out_path}[/]")


@app.command()
def profile(
    model_path: Path = typer.Argument(..., help="Path to a .pt model file"),
):
    """Profile a model's hardware requirements without compressing it."""
    import torch
    import slimnet

    with console.status("Loading and profiling model..."):
        model = torch.load(str(model_path), map_location="cpu")
        dummy = torch.randn(1, 3, 224, 224)
        profiler = slimnet.HardwareProfiler()
        hw = profiler.profile(model, dummy)

    table = Table(title="Hardware Profile", border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Device", hw.device_type.upper())
    table.add_row("GPU", hw.cuda_device_name)
    table.add_row("VRAM Total", f"{hw.vram_total_mb:.0f} MB")
    table.add_row("VRAM Free", f"{hw.vram_free_mb:.0f} MB")
    table.add_row("RAM Total", f"{hw.ram_total_mb:.0f} MB")
    table.add_row("Model Size", f"{hw.model_size_mb:.1f} MB")
    table.add_row("Baseline Latency", f"{hw.baseline_latency_ms:.2f} ms")
    table.add_row("Peak Memory", f"{hw.baseline_memory_mb:.1f} MB")
    console.print(table)


@app.command()
def info():
    """Show SlimNet version and supported targets."""
    import slimnet
    console.print(f"[bold cyan]SlimNet[/] v{slimnet.__version__}")
    console.print("\nSupported targets:")
    for name, preset in slimnet.TARGET_HARDWARE_PRESETS.items():
        console.print(f"  [yellow]{name:12}[/] {preset['description']}")


def _print_report_table(report) -> None:
    """Print a Rich benchmark table from a BenchmarkReport."""
    table = Table(title="Compression Report", border_style="green", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Original", justify="right")
    table.add_column("Compressed", justify="right")
    table.add_column("Improvement", justify="right", style="green")

    table.add_row(
        "Model Size",
        f"{report.original_size_mb:.1f} MB",
        f"{report.compressed_size_mb:.1f} MB",
        f"{report.compression_ratio:.1f}×",
    )
    table.add_row(
        "Latency",
        f"{report.original_latency_ms:.2f} ms",
        f"{report.compressed_latency_ms:.2f} ms",
        f"{report.speedup:.1f}× faster",
    )
    if report.original_accuracy is not None:
        delta = report.accuracy_delta or 0
        color = "red" if delta < -0.02 else "yellow" if delta < 0 else "green"
        table.add_row(
            "Accuracy",
            f"{report.original_accuracy:.4f}",
            f"{report.compressed_accuracy:.4f}",
            f"[{color}]{delta:+.4f}[/]",
        )
    table.add_row(
        "Peak VRAM",
        "—",
        f"{report.peak_vram_mb:.1f} MB",
        "✓" if report.fits_on_target else "✗",
    )

    console.print(table)
    console.print(f"\nTechniques: {', '.join(report.techniques_applied)}")


if __name__ == "__main__":
    app()
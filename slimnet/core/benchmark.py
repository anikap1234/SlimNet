"""
BenchmarkReport and BenchmarkReporter — PRD §6.3
"""
from __future__ import annotations
import time, dataclasses, io, tempfile, os
from typing import Optional
import torch
import torch.nn as nn


@dataclasses.dataclass
class BenchmarkReport:
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    compression_ratio: float = 0.0
    original_latency_ms: float = 0.0
    compressed_latency_ms: float = 0.0
    speedup: float = 0.0
    original_accuracy: Optional[float] = None
    compressed_accuracy: Optional[float] = None
    accuracy_delta: Optional[float] = None
    techniques_applied: list = dataclasses.field(default_factory=list)
    peak_vram_mb: float = 0.0
    fits_on_target: bool = False
    target: str = ""

    def __str__(self) -> str:
        lines = [
            "=" * 52,
            "  SlimNet BenchmarkReport",
            "=" * 52,
            f"  Size      : {self.original_size_mb:.2f} MB → {self.compressed_size_mb:.2f} MB  ({self.compression_ratio:.2f}x)",
            f"  Latency   : {self.original_latency_ms:.2f} ms → {self.compressed_latency_ms:.2f} ms  ({self.speedup:.2f}x speedup)",
        ]
        if self.original_accuracy is not None:
            lines.append(
                f"  Accuracy  : {self.original_accuracy:.4f} → {self.compressed_accuracy:.4f}  "
                f"(delta {self.accuracy_delta:+.4f})"
            )
        lines.append(f"  Peak VRAM : {self.peak_vram_mb:.1f} MB")
        lines.append(f"  Fits      : {'YES' if self.fits_on_target else 'NO'} ({self.target})")
        lines.append(f"  Applied   : {' → '.join(self.techniques_applied) if self.techniques_applied else 'none'}")
        lines.append("=" * 52)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


def _model_size_mb(model: nn.Module) -> float:
    """
    Accurate model size in MB that works for both regular and quantized models.

    strategy: save to a real temp file and measure bytes on disk.
    torch.save on a quantized model writes the actual packed int8 storage,
    giving a true compressed size. state_dict() element_size() returns 1 byte
    for int8 but nelement() is the logical count, not physical packed count,
    causing the near-zero bug we saw.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            tmp_path = f.name
        torch.save(model.state_dict(), tmp_path)
        size_bytes = os.path.getsize(tmp_path)
        os.unlink(tmp_path)
        return size_bytes / (1024 ** 2)
    except Exception:
        # fallback: sum state_dict tensors
        total = sum(
            t.nelement() * t.element_size()
            for t in model.state_dict().values()
        )
        return total / (1024 ** 2)


def _measure_latency(
    model: nn.Module,
    inp: torch.Tensor,
    device: torch.device,
    n: int = 50,
) -> float:
    """Mean inference latency in ms."""
    # Quantized models must stay on CPU
    eval_device = torch.device("cpu")
    model = model.to(eval_device).eval()
    x = inp.to(eval_device)

    # cast input dtype if fp16/bf16
    try:
        for v in model.state_dict().values():
            if v.dtype in (torch.float16, torch.bfloat16):
                x = x.to(v.dtype)
            break
    except Exception:
        pass

    with torch.no_grad():
        for _ in range(5):
            try:
                model(x)
            except Exception:
                break

    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                model(x)
            except Exception:
                break
            times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times) if times else 0.0


def _peak_vram_mb(model: nn.Module, inp: torch.Tensor, device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(inp.to(device))
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


class BenchmarkReporter:

    def compare(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        dummy_input: torch.Tensor,
        target: str,
        target_vram_budget_mb: float,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        techniques_applied: Optional[list] = None,
        n_timed: int = 50,
    ) -> BenchmarkReport:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        orig_size = _model_size_mb(original_model)
        comp_size = _model_size_mb(compressed_model)
        ratio     = orig_size / comp_size if comp_size > 0 else 1.0

        orig_lat = _measure_latency(original_model,  dummy_input, device, n_timed)
        comp_lat = _measure_latency(compressed_model, dummy_input, device, n_timed)
        speedup  = orig_lat / comp_lat if comp_lat > 0 else 1.0

        peak = _peak_vram_mb(compressed_model, dummy_input, device)

        if target == "cpu":
            import psutil
            avail = psutil.virtual_memory().available / (1024 ** 2)
            fits = comp_size < avail * 0.5
        else:
            fits = (peak < target_vram_budget_mb) if peak > 0 else (comp_size < target_vram_budget_mb * 0.8)

        orig_acc = comp_acc = acc_delta = None
        if val_loader is not None:
            orig_acc  = self._evaluate(original_model,   val_loader, device)
            comp_acc  = self._evaluate(compressed_model, val_loader, device)
            acc_delta = comp_acc - orig_acc

        return BenchmarkReport(
            original_size_mb=orig_size,
            compressed_size_mb=comp_size,
            compression_ratio=ratio,
            original_latency_ms=orig_lat,
            compressed_latency_ms=comp_lat,
            speedup=speedup,
            original_accuracy=orig_acc,
            compressed_accuracy=comp_acc,
            accuracy_delta=acc_delta,
            techniques_applied=techniques_applied or [],
            peak_vram_mb=peak,
            fits_on_target=fits,
            target=target,
        )

    @staticmethod
    def _evaluate(model: nn.Module, loader, device: torch.device) -> float:
        eval_device = torch.device("cpu")
        model = model.to(eval_device).eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    break
                x, y = batch[0].to(eval_device), batch[1].to(eval_device)
                try:
                    out = model(x)
                except Exception:
                    break
                out  = out[0] if isinstance(out, tuple) else out
                pred = out.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total   += y.size(0)
        return correct / max(total, 1)
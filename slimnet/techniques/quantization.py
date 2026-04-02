"""
Quantization Module — PRD §5.2
Supports: dynamic int8, static int8, fp16, bf16.
"""
from __future__ import annotations
import copy, logging
from typing import Optional
import torch
import torch.nn as nn
import torch.quantization as tq
from slimnet.configs import QuantConfig

logger = logging.getLogger(__name__)


class QuantizationModule:
    """
    Usage:
        qm = QuantizationModule()
        compressed, info = qm.apply(model, config, dummy_input, calibration_loader)
    """
    def apply(
        self,
        model: nn.Module,
        config: QuantConfig,
        dummy_input: Optional[torch.Tensor] = None,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> tuple[nn.Module, dict]:
        """
        Apply quantization. Returns (quantized_model, info_dict).
        Never modifies original model — works on a deep copy.
        """
        # Ensure model is on CPU (dynamic quantization requires CPU backend)
        try:
            if next(model.parameters()).is_cuda:
                model = model.cpu()
        except StopIteration:
         pass
        if dummy_input is not None:
            dummy_input = dummy_input.cpu()
        mode = config.mode
        logger.info(f"[Quantization] mode={mode}")
        model_copy = copy.deepcopy(model).cpu().eval()

        if mode == "dynamic":
            quantized = self._dynamic(model_copy)
        elif mode == "static":
            if calibration_loader is None:
                logger.warning("[Quantization] static mode needs calibration_loader; falling back to dynamic")
                quantized = self._dynamic(model_copy)
            else:
                quantized = self._static(model_copy, calibration_loader, config.per_channel)
        elif mode == "fp16":
            quantized = model_copy.half()
        elif mode == "bf16":
            quantized = model_copy.to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown quantization mode: {mode!r}")

        # Validate: cosine similarity > threshold (PRD §5.2)
        sim = self._validate(model_copy, quantized, dummy_input) if dummy_input is not None else None
        if sim is not None and sim < config.validation_threshold:
            logger.warning(
                f"[Quantization] cosine similarity {sim:.5f} < threshold "
                f"{config.validation_threshold}. Falling back to dynamic."
            )
            quantized = self._dynamic(copy.deepcopy(model).cpu().eval())
            sim = self._validate(model_copy, quantized, dummy_input)

        info = {"mode": mode, "cosine_similarity": sim}
        logger.info(f"[Quantization] done. cosine_sim={sim}")
        return quantized, info

    def _dynamic(self, model: nn.Module) -> nn.Module:
        return tq.quantize_dynamic(model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8)

    def _static(self, model: nn.Module, loader: torch.utils.data.DataLoader, per_channel: bool) -> nn.Module:
        model.qconfig = tq.get_default_qconfig("fbgemm" if per_channel else "qnnpack")
        tq.prepare(model, inplace=True)
        model.eval()
        n = 0
        with torch.no_grad():
            for batch in loader:
                inp = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(inp.cpu())
                n += 1
                if n >= 128:
                    break
        tq.convert(model, inplace=True)
        return model

    def _validate(self, original: nn.Module, quantized: nn.Module, dummy_input: torch.Tensor) -> float:
        original.eval(); quantized.eval()
        inp = dummy_input.cpu()
        # cast input if model is fp16/bf16
        p_dtype = next(quantized.parameters(), torch.zeros(1)).dtype
        inp_q = inp.to(p_dtype) if p_dtype in (torch.float16, torch.bfloat16) else inp
        with torch.no_grad():
            o1 = original(inp).float().flatten(1)
            o2 = quantized(inp_q).float().flatten(1)
        return torch.nn.functional.cosine_similarity(o1, o2, dim=1).mean().item()
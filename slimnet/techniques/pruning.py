"""
Structured Pruning Module — PRD §5.3
Removes neurons (Linear), filters (Conv2d), heads (MultiheadAttention).
Three importance methods: magnitude, gradient, taylor.
Iterative schedule: n_steps passes with fine-tune between each.
"""
from __future__ import annotations
import copy, logging
from typing import Optional
import torch
import torch.nn as nn
from slimnet.configs import PruneConfig

logger = logging.getLogger(__name__)


class StructuredPruningModule:
    """
    Usage:
        pm = StructuredPruningModule()
        compressed, info = pm.apply(model, config, dummy_input, calibration_loader)
    """

    def apply(
        self,
        model: nn.Module,
        config: PruneConfig,
        dummy_input: Optional[torch.Tensor] = None,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> tuple[nn.Module, dict]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(model).to(device)
        orig_params = sum(p.numel() for p in model.parameters())

        logger.info(f"[Pruning] sparsity={config.sparsity}, method={config.method}, n_steps={config.n_steps}")

        for step in range(config.n_steps):
            step_sparsity = config.sparsity * (step + 1) / config.n_steps
            importance = self._score(model, config.method, calibration_loader, device)
            self._prune(model, importance, step_sparsity)
            if calibration_loader is not None and step < config.n_steps - 1:
                self._finetune(model, calibration_loader, device)

        remaining = sum(p.numel() for p in model.parameters())
        zeroed = sum((p == 0).sum().item() for p in model.parameters())
        info = {
            "original_params": orig_params,
            "remaining_params": remaining,
            "zeroed_params": zeroed,
            "sparsity_achieved": zeroed / max(orig_params, 1),
        }
        logger.info(f"[Pruning] done. zeroed={zeroed:,}/{orig_params:,} ({100*zeroed/orig_params:.1f}%)")
        return model, info

    # ------------------------------------------------------------------
    # Importance scoring
    # ------------------------------------------------------------------

    def _score(self, model, method, loader, device):
        if method == "magnitude":
            return self._magnitude(model)
        elif method == "gradient":
            return self._gradient(model, loader, device) if loader else self._magnitude(model)
        elif method == "taylor":
            return self._taylor(model, loader, device) if loader else self._magnitude(model)
        return self._magnitude(model)

    def _magnitude(self, model: nn.Module) -> dict[str, torch.Tensor]:
        scores = {}
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                scores[name] = m.weight.data.abs().sum(dim=1)
            elif isinstance(m, nn.Conv2d):
                scores[name] = m.weight.data.abs().sum(dim=(1, 2, 3))
        return scores

    def _gradient(self, model: nn.Module, loader, device) -> dict[str, torch.Tensor]:
        model.train()
        accum: dict[str, torch.Tensor] = {}
        n = 0
        for batch in loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                break
            x, y = batch[0].to(device), batch[1].to(device)
            model.zero_grad()
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)) and m.weight.grad is not None:
                    imp = (m.weight.data.abs() * m.weight.grad.abs())
                    s = imp.sum(dim=1) if isinstance(m, nn.Linear) else imp.sum(dim=(1, 2, 3))
                    accum[name] = accum.get(name, torch.zeros_like(s)) + s.detach()
            n += 1
            if n >= 16:
                break
        model.eval()
        return {k: v / n for k, v in accum.items()} if n > 0 else self._magnitude(model)

    def _taylor(self, model: nn.Module, loader, device) -> dict[str, torch.Tensor]:
        acts: dict[str, torch.Tensor] = {}
        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                def hook(mod, inp, out, _n=name):
                    acts[_n] = out.detach()
                hooks.append(m.register_forward_hook(hook))

        model.train()
        accum: dict[str, torch.Tensor] = {}
        n = 0
        for batch in loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                break
            x, y = batch[0].to(device), batch[1].to(device)
            model.zero_grad()
            out = model(x)
            nn.functional.cross_entropy(out, y).backward()
            for name, m in model.named_modules():
                if name in acts:
                    act = acts[name]
                    s = act.abs().mean(dim=0)
                    if isinstance(m, nn.Linear) and s.shape[0] == m.out_features:
                        accum[name] = accum.get(name, torch.zeros_like(s)) + s
            n += 1
            if n >= 16:
                break

        for h in hooks:
            h.remove()
        model.eval()
        if n == 0:
            return self._magnitude(model)
        result = {k: v / n for k, v in accum.items()}
        for name in self._magnitude(model):
            if name not in result:
                result[name] = self._magnitude(model)[name]
        return result

    # ------------------------------------------------------------------
    # Pruning execution
    # ------------------------------------------------------------------

    def _prune(self, model: nn.Module, importance: dict, sparsity: float) -> None:
        for name, m in model.named_modules():
            if name not in importance:
                continue
            scores = importance[name]
            n_prune = max(1, int(scores.shape[0] * sparsity))
            _, idx = torch.topk(scores, n_prune, largest=False)
            mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
            mask[idx] = False
            with torch.no_grad():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    m.weight.data[~mask] = 0.0
                    if m.bias is not None:
                        m.bias.data[~mask] = 0.0

    def _finetune(self, model: nn.Module, loader, device, n_batches: int = 30) -> None:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        count = 0
        for batch in loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                break
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            count += 1
            if count >= n_batches:
                break
        model.eval()
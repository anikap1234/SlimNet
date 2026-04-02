"""
Knowledge Distillation Module — PRD §5.5
Core DL component. Student trained to mimic teacher's output distribution.

Loss (PRD §5.5.1):
    L = α × CrossEntropy(student, hard_labels)
      + β × KL(softmax(student/T) || softmax(teacher/T))

Teacher: frozen in eval() throughout.
Optimizer: AdamW lr=2e-5, cosine LR with 10% warmup, grad clip 1.0.
"""
from __future__ import annotations
import math, logging, copy
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from slimnet.configs import DistillConfig

logger = logging.getLogger(__name__)


class FeatureHook:
    def __init__(self, module: nn.Module):
        self.features: Optional[torch.Tensor] = None
        self._h = module.register_forward_hook(self._hook)
    def _hook(self, m, inp, out):
        self.features = out[0] if isinstance(out, tuple) else out
    def remove(self):
        self._h.remove()


class IntermediateProjection(nn.Module):
    def __init__(self, s_dim: int, t_dim: int):
        super().__init__()
        self.proj = nn.Linear(s_dim, t_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
    def forward(self, x):
        return self.proj(x)


class KnowledgeDistillationModule:

    def train(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: DistillConfig,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        max_accuracy_drop: float = 0.02,
        original_accuracy: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[nn.Module, dict]:

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # PRD §5.5.3: teacher frozen in eval(), gradients fully disabled
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        student = student.to(device).train()

        optimizer = torch.optim.AdamW(
            student.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        total_steps  = len(train_loader) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler    = self._cosine_warmup(optimizer, warmup_steps, total_steps)

        # Mixed precision — CUDA only, disabled to avoid dtype mismatch on T4
        # autocast can cast inputs to fp16 while model weights stay fp32 → RuntimeError
        # Safest: disable autocast, keep everything fp32
        use_amp = False   # was causing HalfTensor/FloatTensor mismatch on T4

        t_hooks, s_hooks, projs = [], [], []
        if config.intermediate:
            t_hooks, s_hooks, projs = self._setup_hooks(teacher, student, device)

        logger.info(
            f"[Distillation] epochs={config.epochs}, T={config.temperature}, "
            f"\u03b1={config.alpha}, \u03b2={config.beta}, device={device}, amp={use_amp}"
        )

        best_state = None
        best_val   = None
        history    = []

        for epoch in range(config.epochs):
            student.train()
            total_loss = 0.0
            n = 0

            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    x      = batch[0].to(device)
                    labels = batch[1].to(device) if len(batch) >= 2 else None
                else:
                    x      = batch.to(device)
                    labels = None

                # Ensure input dtype matches model weights (fp32)
                # This prevents HalfTensor/FloatTensor mismatch on T4
                x = x.float()

                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    t_out    = teacher(x)
                    t_logits = t_out[0] if isinstance(t_out, tuple) else t_out

                s_out    = student(x)
                s_logits = s_out[0] if isinstance(s_out, tuple) else s_out

                loss = self._loss(
                    s_logits, t_logits, labels, config,
                    t_hooks, s_hooks, projs
                )

                loss.backward()
                # PRD §5.5.3: gradient clipping at norm 1.0
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)

                # FIX: optimizer.step() BEFORE scheduler.step()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                n += 1

            avg_loss = total_loss / max(n, 1)
            history.append(avg_loss)
            logger.info(
                f"[Distillation] epoch {epoch+1}/{config.epochs}  loss={avg_loss:.4f}"
            )

            # Early stopping (PRD §5.5.3)
            if val_loader is not None and original_accuracy is not None:
                val_acc = self._evaluate(student, val_loader, device)
                logger.info(
                    f"[Distillation] val_acc={val_acc:.4f} (orig={original_accuracy:.4f})"
                )
                if best_val is None or val_acc > best_val:
                    best_val   = val_acc
                    best_state = {k: v.clone() for k, v in student.state_dict().items()}
                if (original_accuracy - val_acc) > max_accuracy_drop:
                    logger.warning(
                        f"[Distillation] early stop: drop "
                        f"{original_accuracy - val_acc:.4f} > {max_accuracy_drop}"
                    )
                    break

        if best_state is not None:
            student.load_state_dict(best_state)

        for h in t_hooks + s_hooks:
            h.remove()

        student.eval()
        return student, {"loss_history": history, "final_val_acc": best_val}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _loss(self, s_logits, t_logits, labels, config, t_hooks, s_hooks, projs):
        T = config.temperature

        # KL distillation loss — T² scaling preserves gradient magnitude (Hinton 2015)
        s_soft  = F.log_softmax(s_logits / T, dim=-1)
        t_soft  = F.softmax(t_logits / T, dim=-1)
        kl_loss = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T ** 2)

        if labels is not None and config.alpha > 0:
            if labels.dtype in (torch.long, torch.int):
                task_loss = F.cross_entropy(s_logits, labels)
            else:
                task_loss = F.mse_loss(s_logits.squeeze(), labels.float())
            total = config.alpha * task_loss + config.beta * kl_loss
        else:
            total = kl_loss

        if config.intermediate and t_hooks and s_hooks:
            total = total + 0.1 * self._feature_loss(t_hooks, s_hooks, projs)

        return total

    def _feature_loss(self, t_hooks, s_hooks, projs):
        total = torch.tensor(0.0)
        for i in range(min(len(t_hooks), len(s_hooks), len(projs))):
            tf, sf = t_hooks[i].features, s_hooks[i].features
            if tf is None or sf is None:
                continue
            sf_flat = sf.reshape(sf.size(0), -1)
            tf_flat = tf.reshape(tf.size(0), -1)
            try:
                if sf_flat.size(1) != tf_flat.size(1):
                    sf_flat = projs[i](sf_flat)
                total = total + F.mse_loss(sf_flat, tf_flat.detach())
            except Exception:
                pass
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _setup_hooks(self, teacher, student, device):
        def linears(m):
            return [l for l in m.modules() if isinstance(l, nn.Linear)][1:-1]
        tl, sl = linears(teacher), linears(student)
        if not tl or not sl:
            return [], [], []
        n    = min(4, len(tl), len(sl))
        ti   = [int(i * len(tl) / n) for i in range(n)]
        si   = [int(i * len(sl) / n) for i in range(n)]
        th   = [FeatureHook(tl[i]) for i in ti]
        sh   = [FeatureHook(sl[i]) for i in si]
        projs = [
            IntermediateProjection(
                sl[si[i]].out_features, tl[ti[i]].out_features
            ).to(device)
            for i in range(n)
        ]
        return th, sh, projs

    def _cosine_warmup(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return LambdaLR(optimizer, lr_lambda)

    def _evaluate(self, model: nn.Module, loader, device: torch.device) -> float:
        eval_device = torch.device("cpu")
        model = model.to(eval_device).eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    break
                x, y = batch[0].to(eval_device), batch[1].to(eval_device)
                x    = x.float()
                out  = model(x)
                out  = out[0] if isinstance(out, tuple) else out
                pred = out.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total   += y.size(0)
        return correct / max(total, 1)
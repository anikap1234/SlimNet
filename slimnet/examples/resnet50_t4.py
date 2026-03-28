"""
Example 1: ResNet-50 → T4  (PRD §8 Week 6 headline example)

Target benchmark (PRD §9):
    Original:   100 MB,  baseline latency
    Compressed:  18 MB,  4.2× speedup
    Accuracy drop: < 1.5%
    Target device: CPU / T4

Run on Colab T4:
    !pip install slimnet torchvision
    !python examples/resnet50_t4.py
"""

import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import random

import slimnet

# ── 1. Load pretrained ResNet-50 ──────────────────────────────────────────
print("Loading ResNet-50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

original_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {original_params:,}")
print(f"  Size (fp32): {sum(p.numel()*p.element_size() for p in model.parameters())/(1024**2):.1f} MB")

# ── 2. CIFAR-10 calibration data ──────────────────────────────────────────
# ResNet-50 ImageNet input: 224×224. We resize CIFAR-10 for demo.
transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = CIFAR10(root="/tmp/cifar10", train=True, download=True, transform=transform)
val_ds   = CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=transform)

# PRD §5.5: min 512 calibration samples
calib_idx = random.sample(range(len(train_ds)), 1024)
val_idx   = random.sample(range(len(val_ds)),   500)

calib_loader = DataLoader(Subset(train_ds, calib_idx), batch_size=16, num_workers=2)
val_loader   = DataLoader(Subset(val_ds, val_idx),     batch_size=32, num_workers=2)

sample_input = torch.randn(1, 3, 224, 224)

# ── 3. Compress ──────────────────────────────────────────────────────────
print("\nStarting compression pipeline...")
result = slimnet.compress(
    model=model,
    calibration_data=calib_loader,
    target="t4",
    max_accuracy_drop=0.02,
    val_data=val_loader,
    sample_input=sample_input,
    verbose=True,
)

# ── 4. Results ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(result.report)

# ── 5. Verify compressed model works ─────────────────────────────────────
result.model.eval()
with torch.no_grad():
    out = result.model(sample_input)
print(f"\nVerification forward pass: output shape = {out.shape}  ✓")

# ── 6. Save ───────────────────────────────────────────────────────────────
result.save("resnet50_t4_compressed.slimnet")
print("Saved to resnet50_t4_compressed.slimnet")
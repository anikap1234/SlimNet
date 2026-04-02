import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import random

import slimnet

# ── 1. Load smaller model (FASTER) ─────────────────────────────
print("Loading ResNet-18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

original_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {original_params:,}")
print(f"  Size (fp32): {sum(p.numel()*p.element_size() for p in model.parameters())/(1024**2):.1f} MB")

# ── 2. CIFAR-10 (FIXED PATH + SMALLER DATA) ───────────────────
transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

train_ds = CIFAR10(root="data/cifar10", train=True, download=True, transform=transform)
val_ds   = CIFAR10(root="data/cifar10", train=False, download=True, transform=transform)

# 🔥 smaller subset (FAST)
calib_idx = random.sample(range(len(train_ds)), 200)
val_idx   = random.sample(range(len(val_ds)),   200)

# 🔥 FIX: num_workers=0 for Windows
calib_loader = DataLoader(Subset(train_ds, calib_idx), batch_size=16, num_workers=0)
val_loader   = DataLoader(Subset(val_ds, val_idx),     batch_size=32, num_workers=0)

sample_input = torch.randn(1, 3, 224, 224)

# ── 3. Run pipeline ───────────────────────────────────────────
print("\nStarting compression pipeline...")
result = slimnet.compress(
    model=model,
    calibration_data=calib_loader,
    target="cpu",
    max_accuracy_drop=0.02,
    val_data=val_loader,
    sample_input=sample_input,
    verbose=True,
)

# ── 4. Results ────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(result.report)

# ── 5. Verify ────────────────────────────────────────────────
result.model.eval()
with torch.no_grad():
    out = result.model(sample_input)

print(f"\nVerification forward pass: output shape = {out.shape} ✓")

# ── 6. Save ──────────────────────────────────────────────────
result.save("resnet18_compressed.slimnet")
print("Saved to resnet18_compressed.slimnet")
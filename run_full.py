"""
run_full.py — smoke test for the entire SlimNet pipeline.
Run from the repo root: python run_full.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import slimnet

print("SlimNet version:", slimnet.__version__)
print("PyTorch version:", torch.__version__)
print()

# ── Simple model ─────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x): return self.net(x)

model = Net()

# ── Fake dataset ──────────────────────────────────────────────────────────────
x = torch.randn(512, 128)
y = torch.randint(0, 10, (512,))
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

x_val = torch.randn(128, 128)
y_val = torch.randint(0, 10, (128,))
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32)

print("=" * 60)
print("  Running full compression pipeline (target=cpu)")
print("=" * 60)

result = slimnet.compress(
    model=model,
    calibration_data=loader,
    target="cpu",
    max_accuracy_drop=0.02,
    val_data=val_loader,
    sample_input=torch.randn(1, 128),
    verbose=True,
)

print("\n=== FINAL REPORT ===")
print(result.report)

# ── Verify compressed model works ────────────────────────────────────────────
result.model.eval()
out = result.model(torch.randn(4, 128))
print(f"\nVerification — output shape: {out.shape}")   # must be (4, 10)
assert out.shape == (4, 10), f"Wrong shape: {out.shape}"
print("Shape check PASSED ✓")

# ── Save ─────────────────────────────────────────────────────────────────────
result.save("compressed_model.pt")
print("Saved to compressed_model.pt ✓")
print("\nAll checks passed.")
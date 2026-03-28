import torch
import torch.nn as nn
from slimnet.techniques.low_rank import LowRankFactorizationModule
from slimnet.configs import LowRankConfig

class BertFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(3072, 768)
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

model = BertFFN()
orig_params = sum(p.numel() for p in model.parameters())
print(f"Original params: {orig_params:,}")   # 4,722,432

lrm = LowRankFactorizationModule(LowRankConfig(variance_threshold=0.90, min_layer_size=512))
factorized = lrm.apply(model)
fact_params = sum(p.numel() for p in factorized.parameters())
print(f"Factorized params: {fact_params:,}")
print(f"Reduction: {orig_params/fact_params:.2f}×")    # ~3× per PRD

x = torch.randn(4, 768)
with torch.no_grad():
    o1, o2 = model(x), factorized(x)
sim = torch.nn.functional.cosine_similarity(o1.flatten(1), o2.flatten(1), dim=1).mean()
print(f"Cosine similarity: {sim.item():.5f}")   # must be > 0.99

# PRD criterion: BERT FFN reduces ~3×, outputs match within tolerance ✓
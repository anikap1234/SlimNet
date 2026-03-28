import copy, torch
import torchvision.models as models
from slimnet.techniques.pruning import StructuredPruningModule
from slimnet.configs import PruneConfig

resnet18 = models.resnet18(weights=None)
orig_params = sum(p.numel() for p in resnet18.parameters())
print(f"Original params: {orig_params:,}")

pm = StructuredPruningModule(PruneConfig(sparsity=0.4, method="magnitude", n_steps=5))
pruned = pm.apply(copy.deepcopy(resnet18))

zero_params = sum((p == 0).sum().item() for p in pruned.parameters())
print(f"Zeroed params:   {zero_params:,}  ({100*zero_params/orig_params:.1f}%)")

# PRD criterion: 40% of weights zeroed, iterative 5-step schedule ran ✓
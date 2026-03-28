import torch
import torchvision.models as models
from slimnet.techniques.quantization import QuantizationModule
from slimnet.configs import QuantConfig
from slimnet.core.benchmark import measure_model_size
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

resnet18 = models.resnet18(weights=None).eval()
print(f"Original:   {measure_model_size(resnet18):.1f} MB")  # ~45 MB

qm = QuantizationModule(QuantConfig(mode="dynamic"))
q  = qm.apply(resnet18, sample_input=torch.randn(1, 3, 224, 224))
print(f"Quantized:  {measure_model_size(q):.1f} MB")          # ~11 MB


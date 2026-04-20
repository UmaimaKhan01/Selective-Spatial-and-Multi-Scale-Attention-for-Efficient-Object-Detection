import os
import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from models.baseline_detector import get_baseline_model
from models.selective_fpn import get_selective_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use a fixed dummy image at standard COCO eval resolution
dummy_image = [torch.zeros(3, 800, 1067).to(device)]

configs = [
    ("baseline",      get_baseline_model(91)),
    ("spatial_only",  get_selective_model(91, mode="spatial_only")),
    ("scale_only",    get_selective_model(91, mode="scale_only")),
    ("full",          get_selective_model(91, mode="full")),
]

print(f"{'Model':<20} {'GFLOPs':>10} {'Params (M)':>12}")
print("-" * 45)

for name, model in configs:
    model.to(device)
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_image)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        gflops = flops.total() / 1e9
        params = parameter_count(model)[''] / 1e6
    print(f"{name:<20} {gflops:>10.2f} {params:>12.2f}")


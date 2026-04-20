import os, sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.coco_dataset import COCODetection
from models.selective_fpn import get_selective_model

PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch): return tuple(zip(*batch))

val_dataset = COCODetection(
    img_folder=os.path.join(PROJECT_ROOT, 'data', 'coco', 'val2017'),
    ann_file=os.path.join(PROJECT_ROOT, 'data', 'coco', 'annotations', 'instances_val2017.json'),
    transforms=transforms.ToTensor())

loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
N = 500

def measure(ckpt_path, label, mode='spatial_only'):
    model = get_selective_model(num_classes=91, mode=mode)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    level_sums = {str(l): 0.0 for l in range(4)}
    level_counts = {str(l): 0 for l in range(4)}
    gate_maps = {}
    def make_hook(lv):
        def hook(module, inp, out):
            gate_maps[lv] = out.detach().cpu()
        return hook
    for lv, gate in model.spatial_gates.items():
        gate.register_forward_hook(make_hook(lv))
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= N: break
            model([img.to(device) for img in images])
            for lv in gate_maps:
                level_sums[lv] += gate_maps[lv][0, 0].mean().item()
                level_counts[lv] += 1
    means = [level_sums[str(l)] / max(level_counts[str(l)], 1) for l in range(4)]
    print(f'{label}')
    print(f'  P2={means[0]:.4f}  P3={means[1]:.4f}  P4={means[2]:.4f}  P5={means[3]:.4f}')
    print(f'  Overall mean: {sum(means)/4:.4f}\n')
    return means

configs = [
    ('spatial_only scratch  (sp=1e-4)', 'checkpoints/selective_spatial_only_epoch_1.pth', 'spatial_only'),
    ('spatial_only pretrain (sp=1e-4)', 'checkpoints/selective_spatial_only_pretrain_epoch_1.pth', 'spatial_only'),
    ('full scratch          (sp=1e-4)', 'checkpoints/selective_full_epoch_1.pth', 'full'),
]

print(f'Measuring gate activations on {N} val images...\n{"="*55}')
results = {}
for label, ckpt, mode in configs:
    results[label] = measure(os.path.join(PROJECT_ROOT, ckpt), label, mode)

print('Copy these into fig5_gate_levels():')
for label, vals in results.items():
    print(f"  '{label}': {[round(v,3) for v in vals]},")

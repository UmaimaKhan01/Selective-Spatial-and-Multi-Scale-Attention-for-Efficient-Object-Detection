import os
import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets.coco_dataset import COCODetection
from models.baseline_detector import get_baseline_model
from models.selective_fpn import get_selective_model

PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'
device = torch.device('cuda')

def collate_fn(batch):
    return tuple(zip(*batch))

val_dataset = COCODetection(
    img_folder=os.path.join(PROJECT_ROOT, 'data', 'coco', 'val2017'),
    ann_file=os.path.join(PROJECT_ROOT, 'data', 'coco', 'annotations', 'instances_val2017.json'),
    transforms=transforms.ToTensor()
)
loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                    num_workers=0, collate_fn=collate_fn)

N_IMAGES = 500

def measure_detections_and_gate_sparsity(model, label, is_selective=False):
    model.to(device)
    model.eval()

    det_counts = []
    gate_means = []

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, desc=label, total=N_IMAGES)):
            if i >= N_IMAGES:
                break
            images = [img.to(device) for img in images]

            if is_selective:
                outputs, gate_stats = model(images)
                gate_means.append(
                    sum(v.item() for v in gate_stats.values()) / len(gate_stats)
                )
            else:
                outputs = model(images)

            det_counts.append(len(outputs[0]['boxes']))

    avg_dets = sum(det_counts) / len(det_counts)
    avg_gate = sum(gate_means) / len(gate_means) if gate_means else float('nan')
    print(f"{label:<38} detections/img: {avg_dets:6.1f}   mean_gate: {avg_gate:.3f}")

configs = [
    ("baseline",
     get_baseline_model(91), False, None),
    ("spatial_only scratch sp=1e-4",
     get_selective_model(91, mode="spatial_only"), True,
     os.path.join(PROJECT_ROOT, 'checkpoints', 'selective_spatial_only_epoch_1.pth')),
    ("spatial_only pretrain sp=1e-4",
     get_selective_model(91, mode="spatial_only"), True,
     os.path.join(PROJECT_ROOT, 'checkpoints', 'selective_spatial_only_pretrain_epoch_1.pth')),
]

print(f"\n{'Model':<38} {'detections/img':>15}   {'mean_gate':>10}")
print("-" * 68)

for label, model, is_sel, ckpt in configs:
    if ckpt is not None:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state)
    measure_detections_and_gate_sparsity(model, label, is_sel)

import os
import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from datasets.coco_dataset import COCODetection

PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'


def collate_fn(batch):
    return tuple(zip(*batch))


def measure(model, device, root, ckpt_path, model_label="model", num_warmup=20, num_measure=200):
    val_dataset = COCODetection(
        img_folder=os.path.join(root, "val2017"),
        ann_file=os.path.join(root, "annotations", "instances_val2017.json"),
        transforms=transforms.ToTensor()
    )
    data_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []

    data_iter = iter(data_loader)

    # Warmup: run several images without timing to stabilize GPU
    print(f"Warming up {model_label} ...")
    with torch.no_grad():
        for _ in range(num_warmup):
            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                images, _ = next(data_iter)
            images = [img.to(device) for img in images]
            _ = model(images)

    # Timed measurement
    print(f"Measuring {model_label} ...")
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in tqdm(range(num_measure)):
            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                images, _ = next(data_iter)
            images = [img.to(device) for img in images]

            starter.record()
            _ = model(images)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))  # milliseconds

    timings = np.array(timings)
    mean_ms = timings.mean()
    std_ms = timings.std()
    fps = 1000.0 / mean_ms
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"\n=== {model_label} ===")
    print(f"  Mean latency : {mean_ms:.2f} ms  (std {std_ms:.2f} ms)")
    print(f"  FPS          : {fps:.2f}")
    print(f"  Peak GPU mem : {peak_mem_mb:.1f} MB")

    return mean_ms, fps, peak_mem_mb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "spatial_only", "scale_only", "full"])
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pth file")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(PROJECT_ROOT, "data", "coco")

    if args.model == "baseline":
        from models.baseline_detector import get_baseline_model
        model = get_baseline_model(num_classes=91)
    else:
        from models.selective_fpn import get_selective_model
        model = get_selective_model(num_classes=91, mode=args.model)

    measure(model, device, root, args.ckpt,
            model_label=args.model,
            num_warmup=args.warmup,
            num_measure=args.measure)

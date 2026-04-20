import os
import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import time
import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

from models.baseline_detector import get_baseline_model
from datasets.coco_dataset import COCODetection

PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'


def get_transform():
    return transforms.ToTensor()


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    root = os.path.join(PROJECT_ROOT, "data", "coco")
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dataset = COCODetection(
        img_folder=os.path.join(root, "train2017"),
        ann_file=os.path.join(root, "annotations", "instances_train2017.json"),
        transforms=get_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    num_classes = 91
    model = get_baseline_model(num_classes=num_classes)
    model.to(device)

    # Separate backbone and head parameters with different learning rates.
    # Backbone is pretrained — use 10x lower lr to avoid destroying pretrained features.
    # Head (box_predictor, rpn) is randomly initialized — use full lr.
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params,     'lr': 1e-4},
    ], weight_decay=1e-4)

    num_epochs = 2
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 500

    # Warmup then cosine decay
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss += losses.item()

            if i % 500 == 0 and i > 0:
                avg = epoch_loss / (i + 1)
                lr_backbone = optimizer.param_groups[0]['lr']
                lr_head = optimizer.param_groups[1]['lr']
                print(f"  [Epoch {epoch} | Step {i}] avg loss: {avg:.4f} | lr_backbone: {lr_backbone:.2e} | lr_head: {lr_head:.2e}")

        elapsed = time.time() - start
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} done | avg loss: {avg_loss:.4f} | time: {elapsed:.1f}s")

        ckpt_path = os.path.join(ckpt_dir, f"baseline_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete.")

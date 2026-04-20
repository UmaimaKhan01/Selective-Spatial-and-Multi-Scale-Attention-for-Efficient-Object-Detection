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

from models.selective_fpn import get_selective_model
from datasets.coco_dataset import COCODetection

PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'


def get_transform():
    return transforms.ToTensor()


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full",
                        choices=["spatial_only", "scale_only", "full"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--sparsity_weight", type=float, default=1e-4)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--gates_only", action="store_true",
                        help="Freeze all params except gate modules")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Mode:", args.mode)
    print("Sparsity weight:", args.sparsity_weight)
    print("Gates only:", args.gates_only)
    if args.pretrain:
        print("Pretrain ckpt:", args.pretrain)
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
    model = get_selective_model(num_classes=num_classes, mode=args.mode)

    if args.pretrain is not None:
        print("Loading pretrained weights from baseline...")
        baseline_state = torch.load(args.pretrain, map_location="cpu", weights_only=False)
        model_state = model.state_dict()
        matched = {k: v for k, v in baseline_state.items()
                   if k in model_state and model_state[k].shape == v.shape}
        skipped = [k for k in baseline_state if k not in matched]
        model_state.update(matched)
        model.load_state_dict(model_state)
        print(f"Loaded {len(matched)} matched keys, skipped {len(skipped)} gate keys")

    if args.gates_only:
        print("Freezing all params except gates...")
        for name, param in model.named_parameters():
            if 'spatial_gates' in name or 'scale_gates' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        gate_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable gate params: {gate_param_count:,}")

    model.to(device)

    gate_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and ('spatial_gates' in n or 'scale_gates' in n)]

    if args.gates_only:
        optimizer = optim.AdamW(gate_params, lr=1e-3, weight_decay=1e-4)
    else:
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and 'backbone' in n]
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'backbone' not in n
                       and 'spatial_gates' not in n and 'scale_gates' not in n]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params,     'lr': 1e-4},
            {'params': gate_params,     'lr': 1e-4},
        ], weight_decay=1e-4)

    num_epochs = args.epochs
    total_steps = num_epochs * len(train_loader)
    warmup_steps = 500

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    ckpt_name = f"selective_{args.mode}"
    if args.suffix:
        ckpt_name += f"_{args.suffix}"

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [{args.mode}]")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                loss_dict, gate_stats = model(images, targets)
                detection_loss = sum(loss for loss in loss_dict.values())

                sparsity_loss = torch.tensor(0.0, device=device)
                for level_mean in gate_stats.values():
                    sparsity_loss = sparsity_loss + level_mean
                sparsity_loss = sparsity_loss / max(len(gate_stats), 1)

                losses = detection_loss + args.sparsity_weight * sparsity_loss

            scaler.scale(losses).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss += detection_loss.item()

            if i % 500 == 0 and i > 0:
                avg = epoch_loss / (i + 1)
                gate_str = " | ".join(
                    [f"P{int(k)+2}={v.item():.3f}" for k, v in gate_stats.items()]
                )
                lr_cur = optimizer.param_groups[0]['lr']
                print(f"  [Epoch {epoch} | Step {i}] avg loss: {avg:.4f} | lr: {lr_cur:.2e} | gates: {gate_str}")

        elapsed = time.time() - start
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} done | avg loss: {avg_loss:.4f} | time: {elapsed:.1f}s")

        ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete.")

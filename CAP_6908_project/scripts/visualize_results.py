import os
import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from PIL import Image
from datasets.coco_dataset import COCODetection
from models.selective_fpn import get_selective_model

PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 1. mAP vs FPS tradeoff curve ──────────────────────────────────────────────
def plot_tradeoff():
    results = [
        ('Baseline',              0.360, 25.22, 'black',    'o',  200),
        ('Spatial (sp=1e-4)',     0.284, 51.01, 'red',      's',  150),
        ('Spatial (sp=1e-5)',     0.296, 24.46, 'orange',   's',  150),
        ('Scale only',            0.266, 25.03, 'blue',     '^',  150),
        ('Full',                  0.271, 24.34, 'purple',   'D',  150),
        ('Pretrain+Spatial',      0.362, 24.46, 'green',    'P',  200),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, mAP, fps, color, marker, size in results:
        ax.scatter(fps, mAP, c=color, marker=marker, s=size, zorder=5)
        ax.annotate(label, (fps, mAP),
                    textcoords='offset points', xytext=(8, 4),
                    fontsize=9, color=color)

    ax.set_xlabel('FPS (higher is better)', fontsize=12)
    ax.set_ylabel('mAP @ IoU=0.50:0.95 (higher is better)', fontsize=12)
    ax.set_title('Accuracy vs Efficiency Tradeoff', fontsize=14)
    ax.set_xlim(18, 60)
    ax.set_ylim(0.24, 0.39)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(y=0.360, color='black', linestyle=':', alpha=0.4, label='Baseline mAP')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'tradeoff_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ── 2. Per-FPN gate values bar chart ──────────────────────────────────────────
def plot_gate_values():
    # Final epoch gate values from training logs
    models = {
        'Spatial (sp=1e-4)':  [0.357, 0.381, 0.527, 0.756],
        'Spatial (sp=1e-5)':  [0.392, 0.422, 0.546, 0.752],
        'Full (sp=1e-4)':     [0.356, 0.377, 0.536, 0.729],
        'Pretrain+Spatial':   [0.365, 0.387, 0.554, 0.743],
    }
    levels = ['P2', 'P3', 'P4', 'P5']
    x = np.arange(len(levels))
    width = 0.2
    colors = ['red', 'orange', 'purple', 'green']

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (label, vals) in enumerate(models.items()):
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[idx], alpha=0.8)

    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='No gating (1.0)')
    ax.set_xlabel('FPN Level', fontsize=12)
    ax.set_ylabel('Mean Gate Value (lower = more suppression)', fontsize=12)
    ax.set_title('Spatial Gate Activation per FPN Level', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'gate_values_per_level.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ── 3. Gate heatmaps on sample images ─────────────────────────────────────────
def plot_gate_heatmaps(ckpt_path, suffix, n_images=4):
    model = get_selective_model(num_classes=91, mode='spatial_only')
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    val_dataset = COCODetection(
        img_folder=os.path.join(PROJECT_ROOT, 'data', 'coco', 'val2017'),
        ann_file=os.path.join(PROJECT_ROOT, 'data', 'coco', 'annotations', 'instances_val2017.json'),
        transforms=transforms.ToTensor()
    )

    gate_maps = {}
    def make_hook(level):
        def hook(module, input, output):
            gate_maps[level] = output.detach().cpu()
        return hook

    for level, gate in model.spatial_gates.items():
        gate.register_forward_hook(make_hook(level))

    fig, axes = plt.subplots(n_images, 5, figsize=(20, 4 * n_images))
    axes[0][0].set_title('Original', fontsize=11)
    for j, lv in enumerate(['0','1','2','3']):
        axes[0][j+1].set_title(f'Gate P{int(lv)+2}', fontsize=11)

    indices = [0, 10, 50, 100]
    for row, idx in enumerate(indices[:n_images]):
        img_tensor, target = val_dataset[idx]
        with torch.no_grad():
            model([img_tensor.to(device)])

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[row][0].imshow(img_np)
        axes[row][0].axis('off')

        for j, lv in enumerate(['0','1','2','3']):
            if lv in gate_maps:
                gmap = gate_maps[lv][0, 0].numpy()
                axes[row][j+1].imshow(img_np)
                axes[row][j+1].imshow(gmap, cmap='RdYlGn', alpha=0.6,
                                       vmin=0, vmax=1,
                                       extent=[0, img_np.shape[1], img_np.shape[0], 0])
                axes[row][j+1].axis('off')

    plt.suptitle(f'Spatial Gate Heatmaps ({suffix})', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'gate_heatmaps_{suffix}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generating tradeoff curve...')
    plot_tradeoff()

    print('Generating gate value bar chart...')
    plot_gate_values()

    print('Generating gate heatmaps (sp1e4)...')
    plot_gate_heatmaps(
        ckpt_path=os.path.join(PROJECT_ROOT, 'checkpoints', 'selective_spatial_only_epoch_1.pth'),
        suffix='sp1e4'
    )

    print('Generating gate heatmaps (pretrain)...')
    plot_gate_heatmaps(
        ckpt_path=os.path.join(PROJECT_ROOT, 'checkpoints', 'selective_spatial_only_pretrain_epoch_1.pth'),
        suffix='pretrain'
    )

    print('All visualizations saved to:', OUTPUT_DIR)

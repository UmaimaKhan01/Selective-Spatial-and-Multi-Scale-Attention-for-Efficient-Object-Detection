"""
generate_all_figures.py  (v2 - fixed)
======================================
Fixes:
  - Heatmaps: colorbar placed cleanly to the right, no overlap
  - No LaTeX escape characters in matplotlib labels (use _ not \_ )
  - Fig 6: label offsets corrected to prevent overlap
  - Row labels no longer cut off in heatmaps
  - tight_layout replaced with constrained_layout where needed

Run on cluster:
    conda activate gpu_env_evc7
    cd /groups/mli2/CAP_6908_project/CAP_6908_project
    python scripts/generate_all_figures.py
"""

import os, sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from torchvision import transforms

PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'
FIG_DIR      = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

CKPT_SCRATCH  = os.path.join(PROJECT_ROOT, 'checkpoints', 'selective_spatial_only_epoch_1.pth')
CKPT_PRETRAIN = os.path.join(PROJECT_ROOT, 'checkpoints', 'selective_spatial_only_pretrain_epoch_1.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.linestyle':    '--',
    'grid.alpha':        0.4,
})

def save(name):
    for ext in ('pdf', 'png'):
        path = os.path.join(FIG_DIR, f'{name}.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {name}.pdf / .png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Architecture
# ══════════════════════════════════════════════════════════════════════════════
def fig1_architecture():
    print('Fig 1: Architecture...')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.5); ax.axis('off')

    C_BB   = '#4A90D9'
    C_FPN  = '#E67E22'
    C_GATE = '#E63946'
    C_RPN  = '#27AE60'
    C_ROI  = '#8E44AD'

    def box(x, y, w, h, label, sub, color):
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.06', facecolor=color,
            edgecolor='white', linewidth=1.5, alpha=0.92, zorder=3))
        ax.text(x, y + 0.07, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=4)
        if sub:
            ax.text(x, y - 0.25, sub, ha='center', va='center',
                    fontsize=6.5, color='white', alpha=0.88, zorder=4)

    def arrow(x1, y1, x2, y2, ls='solid', color='#555', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color=color,
                            lw=lw, linestyle=ls), zorder=2)

    # Input image
    ax.add_patch(FancyBboxPatch((0.1, 1.75), 0.9, 1.0,
        boxstyle='round,pad=0.05', facecolor='#BDC3C7',
        edgecolor='white', linewidth=1.2, alpha=0.9, zorder=3))
    ax.text(0.55, 2.25, 'Input\nImage', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#2C3E50', zorder=4)
    arrow(1.0, 2.25, 1.35, 2.25)

    box(1.85, 2.25, 0.9, 1.0, 'ResNet-50', 'Backbone', C_BB)
    arrow(2.30, 2.25, 2.65, 2.25)
    box(3.15, 2.25, 0.9, 1.0, 'FPN', 'P2-P5', C_FPN)

    levels_y = [3.70, 3.15, 2.60, 2.05]
    for lv, ly in zip(['P2', 'P3', 'P4', 'P5'], levels_y):
        arrow(3.60, 2.25, 3.85, ly, ls='dashed', color=C_FPN, lw=1.0)
        ax.text(4.22, ly, lv, ha='center', va='center',
                fontsize=7.5, color=C_FPN, fontweight='bold', zorder=4)
        arrow(4.42, ly, 4.58, ly)
        ax.add_patch(FancyBboxPatch((4.58, ly - 0.20), 0.94, 0.40,
            boxstyle='round,pad=0.04', facecolor=C_GATE,
            edgecolor='white', linewidth=1.2, alpha=0.90, zorder=3))
        ax.text(5.05, ly, 'SpatialGate', ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold', zorder=4)
        arrow(5.52, ly, 5.90, ly)

    for ly in levels_y:
        arrow(5.90, ly, 6.70, 2.25)

    box(7.15, 2.25, 0.90, 1.0, 'RPN', 'Proposals', C_RPN)
    arrow(7.60, 2.25, 7.95, 2.25)
    box(8.45, 2.25, 0.90, 1.0, 'RoI Head', 'Box+Class', C_ROI)
    arrow(8.90, 2.25, 9.25, 2.25)

    ax.add_patch(FancyBboxPatch((9.25, 1.75), 0.65, 1.0,
        boxstyle='round,pad=0.05', facecolor='#BDC3C7',
        edgecolor='white', linewidth=1.2, alpha=0.9, zorder=3))
    ax.text(9.575, 2.25, 'Output', ha='center', va='center',
            fontsize=7.0, fontweight='bold', color='#2C3E50', zorder=4)

    ax.annotate(r'Sparsity loss: $\lambda \cdot \bar{g}_l$',
        xy=(5.05, 1.00), fontsize=8, color=C_GATE, ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDECEA',
                  edgecolor=C_GATE, linewidth=0.8, alpha=0.85))

    ax.set_title('SelectiveFPNFasterRCNN Architecture',
                 fontsize=12, fontweight='bold', pad=6, color='#2C3E50')

    patches = [
        mpatches.Patch(color=C_BB,   label='ResNet-50 Backbone'),
        mpatches.Patch(color=C_FPN,  label='Feature Pyramid Network'),
        mpatches.Patch(color=C_GATE, label='Spatial Gate'),
        mpatches.Patch(color=C_RPN,  label='RPN'),
        mpatches.Patch(color=C_ROI,  label='RoI Head'),
    ]
    ax.legend(handles=patches, loc='lower center', ncol=5,
              fontsize=7, framealpha=0.92, edgecolor='#ccc',
              bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    save('fig1_architecture')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 & 3 — Heatmaps  (fixed layout: colorbar right, no overlap, no cutoff)
# ══════════════════════════════════════════════════════════════════════════════
def fig_heatmaps(ckpt_path, fig_name, title_suffix):
    print(f'{fig_name}: Heatmaps ({title_suffix})...')
    from models.selective_fpn import get_selective_model
    from datasets.coco_dataset import COCODetection

    model = get_selective_model(num_classes=91, mode='spatial_only')
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()

    val_dataset = COCODetection(
        img_folder=os.path.join(PROJECT_ROOT, 'data', 'coco', 'val2017'),
        ann_file=os.path.join(PROJECT_ROOT, 'data', 'coco', 'annotations',
                              'instances_val2017.json'),
        transforms=transforms.ToTensor())

    gate_maps = {}
    def make_hook(level):
        def hook(module, inp, out):
            gate_maps[level] = out.detach().cpu()
        return hook
    for level, gate in model.spatial_gates.items():
        gate.register_forward_hook(make_hook(level))

    sample_indices = [0, 50, 200, 500]
    n_rows = len(sample_indices)
    n_img_cols = 5   # original + P2 P3 P4 P5

    # Use GridSpec: 5 image columns + 1 thin colorbar column
    fig = plt.figure(figsize=(13.5, 3.2 * n_rows))
    fig.suptitle(f'Spatial Gate Heatmaps — {title_suffix}',
                 fontsize=13, fontweight='bold', y=1.003)

    gs = GridSpec(
        n_rows, n_img_cols + 1,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 0.05],
        wspace=0.03,
        hspace=0.08,
        left=0.09,    # room for row labels
        right=0.92,
        top=0.95,
        bottom=0.02,
    )

    col_titles = ['Original', 'Gate P2\n(finest)', 'Gate P3',
                  'Gate P4', 'Gate P5\n(coarsest)']

    for row, idx in enumerate(sample_indices):
        img_t, _ = val_dataset[idx]
        with torch.no_grad():
            model([img_t.to(device)])
        img_np = np.clip(img_t.permute(1, 2, 0).numpy(), 0, 1)
        H, W   = img_np.shape[:2]

        for col in range(n_img_cols):
            ax = fig.add_subplot(gs[row, col])

            if col == 0:
                ax.imshow(img_np)
                # Row label outside the axis, left side
                ax.set_ylabel(f'S{row + 1}', fontsize=9,
                              rotation=0, labelpad=28, va='center', fontweight='bold')
            else:
                lv = str(col - 1)
                ax.imshow(img_np)
                if lv in gate_maps:
                    gmap = gate_maps[lv][0, 0].numpy()
                    ax.imshow(gmap, cmap='RdYlGn', alpha=0.55,
                              vmin=0, vmax=1,
                              extent=[0, W, H, 0],
                              interpolation='bilinear')

            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            # Column titles on first row only
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9,
                             fontweight='bold', pad=4)

    # Colorbar occupies the last column, spanning all rows
    cbar_ax = fig.add_subplot(gs[:, n_img_cols])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Gate activation\n(0=suppressed, 1=active)',
                   fontsize=8, rotation=270, labelpad=16)
    cbar.ax.tick_params(labelsize=7)

    save(fig_name)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Gate vs Latency
# ══════════════════════════════════════════════════════════════════════════════
def fig4_gate_latency():
    print('Fig 4: Gate vs latency...')
    data = [
        ('Baseline\n(no gating)', 1.000, 39.66, 'o', '#333333', 160),
        ('Pretrain\nλ=1e-4', 0.827, 40.88, 'P', '#4CAF50', 160),
        ('Scratch\nλ=1e-4',  0.487, 19.60, 's', '#E63946', 160),
    ]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    xs = [d[1] for d in data]; ys = [d[2] for d in data]
    z  = np.polyfit(xs, ys, 1)
    xfit = np.linspace(0.45, 1.05, 100)
    ax.plot(xfit, np.polyval(z, xfit), '--', color='#BBBBBB',
            linewidth=1.3, zorder=1)

    for label, gate, lat, marker, color, size in data:
        ax.scatter(gate, lat, marker=marker, color=color,
                   s=size, zorder=5, edgecolors='white', linewidths=1.2)

    label_offsets = {
        'Baseline\n(no gating)': (+0.02, +1.5),
        'Pretrain\nλ=1e-4': (+0.02, +1.5),
        'Scratch\nλ=1e-4':  (+0.02, +1.0),
    }
    for label, gate, lat, marker, color, size in data:
        dx, dy = label_offsets[label]
        ax.annotate(label, xy=(gate, lat),
                    xytext=(gate + dx, lat + dy),
                    fontsize=9, color=color,
                    ha='left', va='bottom',
                    multialignment='center',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='none', alpha=0.82))

    ax.annotate('', xy=(0.487, 19.60), xytext=(1.000, 39.66),
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.8))
    ax.text(0.73, 26.0, '51% suppression\n\u2192 2\u00d7 speedup',
            fontsize=8.5, color='#E63946', ha='center', style='italic')

    ax.set_xlabel('Mean Gate Activation  (lower = more suppression)', fontsize=10)
    ax.set_ylabel('Wall-Clock Latency  (ms)', fontsize=10)
    ax.set_title('Gate Activation vs. Wall-Clock Latency', fontsize=11, pad=8)
    ax.set_xlim(0.37, 1.16)
    ax.set_ylim(11, 51)

    plt.tight_layout(pad=1.0)
    save('fig4_gate_latency_scatter')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Per-Level Gate Bars  (plain text, no LaTeX escapes)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_gate_levels():
    print('Fig 5: Per-level gate bars...')
    gate_data = {
        'spatial_only (scratch, λ=1e-4)':  [0.343, 0.359, 0.521, 0.723],
        'spatial_only (pretrain, λ=1e-4)': [0.709, 0.791, 0.862, 0.947],
        'full (scratch, λ=1e-4)':          [0.927, 0.778, 0.847, 0.979],
    }
    levels  = ['P2', 'P3', 'P4', 'P5']
    colors  = ['#E63946', '#4CAF50', '#6A4C93']
    n       = len(gate_data)
    x       = np.arange(4)
    w       = 0.24
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    for idx, (label, vals) in enumerate(gate_data.items()):
        bars = ax.bar(x + offsets[idx], vals, w,
                      label=label, color=colors[idx],
                      alpha=0.88, edgecolor='white', linewidth=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                    f'{v:.2f}', ha='center', va='bottom',
                    fontsize=7.5, color='#333')

    ax.axhline(y=1.0, color='#888', linestyle='--', linewidth=1.2,
               label='No gating (1.0)', zorder=2)

    ax.set_xlabel('FPN Level', fontsize=11)
    ax.set_ylabel('Mean Gate Activation\n(lower = more suppression)', fontsize=10)
    ax.set_title('Spatial Gate Activation per FPN Level', fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=11)
    ax.set_ylim(0, 1.22)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='upper left',
              framealpha=0.92, edgecolor='#ccc')

    plt.tight_layout(pad=1.0)
    save('fig5_gate_levels')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Tradeoff Scatter  (plain text, fixed label positions)
# ══════════════════════════════════════════════════════════════════════════════
def fig6_tradeoff():
    print('Fig 6: Tradeoff scatter...')
    models = [
        ('Baseline',                                  0.360, 25.22, 'o', '#333333', 130, (+1.5, +0.003)),
        ('scale_only',                                0.266, 25.03, '^', '#2196F3', 110, (+1.5, -0.009)),
        ('full',                                      0.271, 24.34, 'D', '#9C27B0', 110, (-10.0, -0.010)),
        ('spatial_only\n(scratch λ=1e-4)',       0.284, 51.01, 's', '#E63946', 140, (+1.5, +0.003)),
        ('spatial_only\n(pretrain λ=1e-4)',      0.362, 24.46, 'P', '#4CAF50', 140, (+1.5, +0.005)),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.axhline(y=0.360, color='#999', linestyle=':', linewidth=1.3, zorder=1)

    for label, mAP, fps, marker, color, size, _ in models:
        ax.scatter(fps, mAP, marker=marker, color=color,
                   s=size, zorder=5, edgecolors='white', linewidths=1.0)

    for label, mAP, fps, marker, color, size, (dx, dy) in models:
        ax.annotate(label,
                    xy=(fps, mAP),
                    xytext=(fps + dx, mAP + dy),
                    fontsize=8.5, color=color,
                    ha='left', va='center',
                    multialignment='center',
                    bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                              edgecolor='none', alpha=0.82))

    ax.annotate('', xy=(51.01, 0.284), xytext=(25.22, 0.360),
                arrowprops=dict(arrowstyle='->', color='#E63946',
                                lw=1.6, linestyle='dashed'))
    ax.text(38.5, 0.324, '2\u00d7 speedup\n\u22127.6 mAP',
            fontsize=8.5, color='#E63946', ha='center', style='italic')

    ax.text(59, 0.362, 'Baseline mAP', fontsize=7.5,
            color='#999', va='bottom', ha='right')

    ax.set_xlabel('FPS  (higher is better)', fontsize=11)
    ax.set_ylabel('mAP @ IoU=0.50:0.95\n(higher is better)', fontsize=10)
    ax.set_title('Accuracy\u2013Efficiency Tradeoff on COCO 2017 val',
                 fontsize=12, pad=8)
    ax.set_xlim(10, 65)
    ax.set_ylim(0.225, 0.400)
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
               markersize=8, label=lbl.replace('\n', ' '))
        for lbl, _, _, m, c, _, _ in models
    ] + [Line2D([0], [0], linestyle=':', color='#999',
                label='Baseline mAP (0.360)')]

    ax.legend(handles=legend_elements, fontsize=8.5, loc='lower right',
              framealpha=0.94, edgecolor='#ccc')

    plt.tight_layout(pad=1.0)
    save('fig6_tradeoff')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'\nGenerating all figures -> {FIG_DIR}\n{"="*50}')
    fig1_architecture()
    fig_heatmaps(CKPT_SCRATCH,  'fig2_heatmaps_scratch',  'scratch, lambda=1e-4')
    fig_heatmaps(CKPT_PRETRAIN, 'fig3_heatmaps_pretrain', 'pretrain, lambda=1e-4')
    fig4_gate_latency()
    fig5_gate_levels()
    fig6_tradeoff()
    print(f'\n{"="*50}')
    print(f'All figures saved to: {FIG_DIR}')
    for f in sorted(os.listdir(FIG_DIR)):
        if f.endswith('.pdf') or f.endswith('.png'):
            sz = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
            print(f'  {f:<42} {sz:6.1f} KB')
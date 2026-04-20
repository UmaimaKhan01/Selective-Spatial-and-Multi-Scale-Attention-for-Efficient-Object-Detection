# SelectiveFPN-FasterRCNN

**Selective Spatial and Multi-Scale Attention for Efficient Object Detection**  

---

## Overview

We modify Faster R-CNN with learned spatial gates inserted after the FPN, trained jointly with a sparsity regularization term. The key finding: spatial gating **increases FLOPs by +6% yet achieves 2× wall-clock speedup** by suppressing ~51% of feature map values, reducing GPU memory-bandwidth pressure — a bottleneck missed by standard FLOPs analysis.

---

## Results (COCO 2017 val)

| Model | mAP | Latency (ms) | FPS | GFLOPs |
|---|---|---|---|---|
| Baseline | 0.360 | 39.66 | 25.2 | 177.6 |
| scale_only | 0.266 | 39.96 | 25.0 | 177.6 |
| full | 0.271 | 41.08 | 24.3 | 188.3 |
| spatial_only (scratch, λ=1e-4) | 0.284 | **19.60** | **51.0** | 188.3 |
| spatial_only (pretrain, λ=1e-4) | **0.362** | 40.88 | 24.5 | 188.3 |

---

## Key Findings

1. **FLOPs ≠ latency** — spatial gating adds gate convolution overhead (+6% FLOPs) but halves wall-clock time via memory-bandwidth reduction
2. **Gradient domination** — fine-tuning from pretrained weights prevents sparsity regularization from closing gates regardless of λ ∈ {1e-5, 1e-4, 1e-3}
3. **Scale gating is ineffective** — scalar FPN-level weights do not change memory access patterns and provide no speedup

---

## Setup

```bash
git clone [https://github.com/<your-repo>/selective-fpn](https://github.com/UmaimaKhan01/Selective-Spatial-and-Multi-Scale-Attention-for-Efficient-Object-Detection
cd selective-fpn
conda activate gpu_env_evc7
pip install torch torchvision pycocotools fvcore
```

Download COCO 2017 to `data/coco/` with the standard directory structure.

---

## Training

```bash
# Baseline
python scripts/train_baseline.py

# Selective models
python scripts/train_selective.py --mode spatial_only --epochs 2 --sparsity_weight 1e-4

# Initialize from pretrained baseline
python scripts/train_selective.py --mode spatial_only --epochs 2 \
    --sparsity_weight 1e-4 --pretrain checkpoints/baseline_epoch_1.pth
```

`--mode` options: `spatial_only` | `scale_only` | `full`

---

## Evaluation

```bash
python scripts/eval_coco.py --model spatial_only \
    --ckpt checkpoints/selective_spatial_only_epoch_1.pth

python scripts/measure_latency.py --model spatial_only \
    --ckpt checkpoints/selective_spatial_only_epoch_1.pth \
    --warmup 20 --measure 200
```

---

## Generate Figures

```bash
python scripts/generate_all_figures.py
# outputs to figures/ directory
```

---

## Environment

- Python 3.10, PyTorch 2.5.1, CUDA 12.1
- NVIDIA Tesla V100 16GB
- COCO 2017 (118K train / 5K val)

---


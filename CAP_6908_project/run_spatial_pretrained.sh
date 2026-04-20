#!/bin/bash
#SBATCH --partition=normal

#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=spatial_pretrain
#SBATCH --output=/groups/mli2/CAP_6908_project/CAP_6908_project/logs/spatial_pretrain_%j.log

source ~/.bashrc
conda activate gpu_env_evc7
cd /groups/mli2/CAP_6908_project/CAP_6908_project
python scripts/train_selective.py --mode spatial_only --epochs 2 --sparsity_weight 1e-4 --pretrain checkpoints/baseline_epoch_1.pth --suffix pretrain
python scripts/eval_coco.py --model spatial_only --ckpt checkpoints/selective_spatial_only_pretrain_epoch_1.pth
python scripts/measure_latency.py --model spatial_only --ckpt checkpoints/selective_spatial_only_pretrain_epoch_1.pth --warmup 20 --measure 200

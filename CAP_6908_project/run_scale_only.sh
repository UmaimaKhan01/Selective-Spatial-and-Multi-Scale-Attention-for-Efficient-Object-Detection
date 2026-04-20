#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodelist=evc1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=scale_only
#SBATCH --output=/groups/mli2/CAP_6908_project/CAP_6908_project/logs/scale_only_%j.log

source ~/.bashrc
conda activate gpu_env2
cd /groups/mli2/CAP_6908_project/CAP_6908_project
python scripts/train_selective.py --mode scale_only --epochs 2
python scripts/eval_coco.py --model scale_only --ckpt checkpoints/selective_scale_only_epoch_1.pth
python scripts/measure_latency.py --model scale_only --ckpt checkpoints/selective_scale_only_epoch_1.pth --warmup 20 --measure 200

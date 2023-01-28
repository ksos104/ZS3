#!/bin/bash
#SBATCH -J ZS3
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o logs/stdout_%j.txt
#SBATCH -e logs/stderr_%j.txt
#SBATCH --gres=gpu

CUDA_VISIBLE_DEVICES=2

python train_net.py \
  --config-file configs/pascal_voc/zegformer_R101_bs32_10k_vit16_voc.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 2
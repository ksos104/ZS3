#!/bin/bash
#SBATCH -J ZS3
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o logs/stdout_%j.txt
#SBATCH -e logs/stderr_%j.txt
#SBATCH --gres=gpu:4

CUDA_VISIBLE_DEVICES=0,1,2,3
python train_net.py \
  --config-file configs/pascal_voc/dino_clip_bs32_10k_vit_small_voc.yaml \
  --num-gpus 4 SOLVER.IMS_PER_BATCH 32 OUTPUT_DIR ./output_dino_vit_small_voc32_self_attn_dec_nh1_noAttnConv_mean \
  DINO.THRESHOLD 0.3 DINO.NUM_ITER 10
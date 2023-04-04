import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL

gt_root = '/mnt/server14_hard0/msson/ZegFormer/results_vis_gt_voc'
baseline_root = '/mnt/server14_hard0/msson/ZegFormer/results_vis_voc'
nh6_root = 'results_vis_dino_vit_small_nh6'
head_max_root = 'results_vis_dino_vit_small_head_max'
matcher_sum_norm_root = 'results_vis_dino_vit_small_matcher_sum_norm'

save_path = 'vis_comparison'
os.makedirs(save_path, exist_ok=True)

tf = transforms.ToTensor()

img_list = os.listdir(gt_root)

for idx, img_name in enumerate(img_list):
    if os.path.isdir(os.path.join(gt_root, img_name)):
        continue
    
    gt_img = tf(PIL.Image.open(os.path.join(gt_root, img_name)))
    baseline_img = tf(PIL.Image.open(os.path.join(baseline_root, img_name)))
    nh6_img = tf(PIL.Image.open(os.path.join(nh6_root, img_name)))
    head_max_img = tf(PIL.Image.open(os.path.join(head_max_root, img_name)))
    matcher_sum_norm_img = tf(PIL.Image.open(os.path.join(matcher_sum_norm_root, img_name)))
    
    img_seq = torch.stack([gt_img, baseline_img, nh6_img, head_max_img, matcher_sum_norm_img], dim=0)
    
    grid_img = torchvision.utils.make_grid(img_seq, nrow=img_seq.shape[0])
    
    torchvision.utils.save_image(grid_img, os.path.join(save_path, img_name))
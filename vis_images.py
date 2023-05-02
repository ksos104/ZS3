import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL

root_list = [
    '/mnt/server14_hard0/msson/ZegFormer/results_vis_gt_voc',
    '/mnt/server14_hard0/msson/ZegFormer/results_vis_voc',
    # 'results_vis_dino_vit_small_nh6',
    # 'results_vis_dino_vit_small_head_max',
    # 'results_vis_dino_vit_small_matcher_sum_norm',
    # 'results_vis_dino_vit_small_self_attn',
    # 'results_vis_dino_vit_small_self_attn',
    'results_vis_dino_vit_small_self_attn_dec_nh1_sm',
    'results_vis_dino_vit_small_self_attn_dec_nh1_noAttnConv'
]

save_path = 'vis_comparison'
os.makedirs(save_path, exist_ok=True)

tf = transforms.ToTensor()

img_list = os.listdir(root_list[0])

for idx, img_name in enumerate(img_list):
    if os.path.isdir(os.path.join(root_list[0], img_name)):
        continue
    
    img_list = []
    for i in range(len(root_list)):
        img_list.append(tf(PIL.Image.open(os.path.join(root_list[i], img_name))))
        
    img_seq = torch.stack(img_list, dim=0)
    
    grid_img = torchvision.utils.make_grid(img_seq, nrow=img_seq.shape[0])
    
    torchvision.utils.save_image(grid_img, os.path.join(save_path, img_name))
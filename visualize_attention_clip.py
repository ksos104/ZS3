# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from dino_clip import vision_transformer as vits
from dino_clip.third_party import clip

from dino_clip.modeling.heads.dino_decoder import DINODecoder


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    # print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./attn_vis', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    '''
        CLIP
    '''
    ## CLIP init
    def zeroshot_classifier(classnames, templates, clip_modelp):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = []
                    for template in templates:
                        for cls_split in classname_splits:
                            texts.append(template.format(cls_split))
                else:
                    texts = [template.format(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize, shape: [48, 77]
                class_embeddings = clip_modelp.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    class_json = '/mnt/server14_hard1/msson/datasets/zs3_datasets/VOCZERO/all_classnames.json'
    clip_pretrained = 'ViT-B/16'
    prompt_templates = ['A photo of a {} in the scene',]
    import json
    with open(class_json, 'r') as f_in:
            class_texts = json.load(f_in)
    clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False)
    
    features_test = zeroshot_classifier(class_texts, prompt_templates, clip_model).permute(1, 0).float()

    ## CLIP score
    clip_img = nn.functional.interpolate(img.to(device), (224,224))
    image_features = clip_model.encode_image(clip_img)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.exp()
    
    cls_score = logit_scale * image_features @ features_test.clone().detach().t().half()
    # print("image0: ", class_texts[cls_score.argmax()])

    mode = 'ref_fusion'        ## decoder / ref_fusion

    if mode  == 'decoder':
        ## DINO decoder
        dino_decoder = DINODecoder(
            norm = "GN"
        ).to(device)
        model_dict = dino_decoder.state_dict()
        loaded_dict = torch.load('output_dino_vit_small_voc32_self_attn_dec_nh1_noAttnConv/model_0004999.pth')['model']
        
        for k, v in loaded_dict.items():
            if 'dino_decoder' in k:
                new_k = '.'.join(k.split('.')[1:])
                model_dict[new_k] = loaded_dict[k]
        
        num_iter = 10
        bs = img.shape[0]
        remains = torch.ones((bs,1,3600)).to(device)
        th_attn = torch.zeros((bs,6,3600)).to(device)
        th_attn_sum = torch.zeros((bs,1,3600)).to(device)
        for i in range(num_iter):
            features, attentions = model.get_last_selfattention(img.to(device))
            features = features[:,1:,:]

            bs = attentions.shape[0]
            nh = attentions.shape[1] # number of head

            # we keep only the output patch attention
            # attentions = attentions[:, :, 0, 1:].reshape(bs, nh, -1)
            
            ## self-attention map                
            cls_attentions = attentions[:, :, 0, 1:].reshape(bs, nh, -1)
            idx_attentions = torch.argmax(cls_attentions, dim=-1)
            attentions = attentions[:, :, 1:, 1:]
            # attentions = torch.gather(attentions, dim=2, index=idx_attentions)
            temp_attentions = []
            for i_bs in range(bs):
                for i_head in range(nh):
                    temp_attentions.append(attentions[i_bs, i_head, idx_attentions[i_bs, i_head], :])
            attentions = torch.stack(temp_attentions).reshape(bs, nh, -1)
            
            '''
                Decoding
            '''
            features, attentions = dino_decoder(features, attentions)
            ## Select max attention
            attn_max = torch.max(attentions, dim=-1)[0]
            max_idx = torch.max(attn_max, dim=-1)[1]
            temp_attentions = []
            for i_bs in range(bs):
                temp_attentions.append(attentions[i_bs, max_idx[i_bs], ...])
            attentions = torch.stack(temp_attentions).reshape(bs, 1, -1)
            ## Mean attentions
            # attentions = attentions.mean(dim=1).reshape(bs, 1, -1)
            nh = 1
                    
            if args.threshold is not None:
                if i == num_iter - 1:
                    remains -= th_attn_sum
                    th_attn = remains.repeat([1,nh,1])
                    assert remains.min() >= 0
                    
                    # th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                    # # interpolate
                    # th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()                
                else:
                    remains -= th_attn_sum
                    # we keep only a certain percentage of the mass
                    val, idx = torch.sort(attentions, dim=2)
                    val /= torch.sum(val, dim=2, keepdim=True)
                    cumval = torch.cumsum(val, dim=2)
                    th_attn = cumval > (1 - args.threshold)
                    idx2 = torch.argsort(idx, dim=2)
                    # for head in range(nh):
                    #     th_attn[head] = th_attn[head][idx2[head]]
                    th_attn = torch.gather(th_attn, 2, idx2).float()                
                    
                    th_attn = th_attn * remains
                th_attn_sum = th_attn.sum(dim=1).unsqueeze(dim=1)
                
                th_attn_sum = torch.where(th_attn_sum==0, th_attn_sum, torch.tensor([1.],device=device))
                th_attn_imgs = th_attn_sum.reshape(bs, 1, w_featmap, h_featmap)
                th_attn_imgs = nn.functional.interpolate(th_attn_imgs, scale_factor=args.patch_size, mode="nearest")
                
                next_img = img * (1-th_attn_imgs.cpu())
                
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                # interpolate
                th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
                
            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
            
            # save attentions heatmaps
            os.makedirs(args.output_dir, exist_ok=True)
            torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img"+"_iter"+str(i)+".png"))
            for j in range(nh):
                fname = os.path.join(args.output_dir, "attn-head" + str(j) + "_iter" + str(i) + ".png")
                plt.imsave(fname=fname, arr=attentions[j], format='png')
                print(f"{fname} saved.")

            if args.threshold is not None:
                image = skimage.io.imread(os.path.join(args.output_dir, "img"+"_iter"+str(i)+".png"))
                for j in range(nh):
                    display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) + "_iter"+str(i)+".png"), blur=False)
            
            img = next_img
    elif mode == 'ref_fusion':
        '''
            Use refined fusion maps
        '''
        from torch.nn import functional as F
        
        cam_head = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1).to(device)
        
        
        num_iter = 10
        bs = img.shape[0]
        remains = torch.ones((bs,1,3600)).to(device)
        th_attn = torch.zeros((bs,6,3600)).to(device)
        th_attn_sum = torch.zeros((bs,1,3600)).to(device)
        th_attn_imgs = torch.zeros((bs,1,60,60)).to(device)
        
        for i in range(num_iter):
            features, attentions = model.get_last_selfattention(img.to(device))
            features = features[:,1:,:]                 ## features.shape = [bs, 3600, 384]
            
            '''
                feature to CAM
            '''
            features = features.reshape(bs, 60, 60, features.shape[-1])     ## features.shape = [bs, 60, 60, 384]
            features = features.permute(0, 3, 1, 2)
            features = features.contiguous()
            features = cam_head(features)
            
            features = features.detach().clone()
            features = F.relu(features)
            
            attentions = torch.mean(attentions, dim=1, keepdim=True)
            # attentions = torch.max(attentions, dim=1, keepdim=True)[0]
                        
            cls_attentions = attentions[:, :, 0, 1:]
            patch_attentions = attentions[:, :, 1:, 1:]
            
            cls_attn_maps = cls_attentions.unsqueeze(dim=2).reshape(bs, 1, 60, 60)
            
            cams = cls_attn_maps * features
            
            ref_fusion_maps = torch.matmul(patch_attentions, cams.reshape(bs, features.shape[1], -1, 1)).reshape(bs, features.shape[1], 60, 60)
            features = ref_fusion_maps.reshape(bs, -1, 3600).permute(0,2,1)
            
            bs = attentions.shape[0]
            nh = attentions.shape[1] # number of head
            
            # we keep only the output patch attention
            # attentions = attentions[:, :, 0, 1:].reshape(bs, nh, -1)
            
            ## self-attention map
            idx_attentions = torch.argmax(cls_attentions, dim=-1)
            temp_attentions = []
            for i_bs in range(bs):
                for i_head in range(nh):
                    temp_attentions.append(patch_attentions[i_bs, i_head, idx_attentions[i_bs, i_head], :])
            attentions = torch.stack(temp_attentions).reshape(bs, nh, -1)
            
            ## patch_attn (matmul) cls_attn
            # attentions = torch.matmul(patch_attentions, cls_attentions.unsqueeze(-1)).squeeze(-1)
            
            '''
                Decoding
            '''
            # features, attentions = self.dino_decoder(features, attentions)
            # ## Select max attention
            attn_max = torch.max(attentions, dim=-1)[0]
            max_idx = torch.max(attn_max, dim=-1)[1]
            temp_attentions = []
            for i_bs in range(bs):
                temp_attentions.append(attentions[i_bs, max_idx[i_bs], ...])
            attentions = torch.stack(temp_attentions).reshape(bs, 1, -1)
            # ## Mean attentions
            # # attentions = attentions.mean(dim=1).reshape(bs, 1, -1)
            # nh = 1
            
            attentions = cls_attentions * attentions
            # attentions = attentions * (1-th_attn_imgs.reshape(bs,1,3600))

            if i == num_iter - 1:
                remains -= th_attn_sum
                th_attn = remains.repeat([1,nh,1])
                assert remains.min() >= 0
            else:
                remains -= th_attn_sum
                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions, dim=2)
                val /= torch.sum(val, dim=2, keepdim=True)
                cumval = torch.cumsum(val, dim=2)
                th_attn = cumval > (1 - args.threshold)
                idx2 = torch.argsort(idx, dim=2)
                # for head in range(nh):
                    # th_attn[:,head] = th_attn[:,head][idx2[head]]
                th_attn = torch.gather(th_attn, 2, idx2).float()
                # Eliminate elements chosen before
                th_attn = th_attn * remains
            th_attn_sum = th_attn.sum(dim=1).unsqueeze(dim=1)
            
            th_attn_sum = torch.where(th_attn_sum==0, th_attn_sum, torch.tensor([1.],device=device))
            th_attn_imgs = th_attn_sum.reshape(bs, 1, w_featmap, h_featmap)
            th_attn_imgs_orig = nn.functional.interpolate(th_attn_imgs, scale_factor=args.patch_size, mode="nearest")
            
            next_img = img * (1-th_attn_imgs_orig.cpu())
                
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
                
            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
                
            # save attentions heatmaps
            os.makedirs(args.output_dir, exist_ok=True)
            torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img"+"_iter"+str(i)+".png"))
            for j in range(nh):
                fname = os.path.join(args.output_dir, "attn-head" + str(j) + "_iter" + str(i) + ".png")
                plt.imsave(fname=fname, arr=attentions[j], format='png')
                # print(f"{fname} saved.")

            if args.threshold is not None:
                image = skimage.io.imread(os.path.join(args.output_dir, "img"+"_iter"+str(i)+".png"))
                for j in range(nh):
                    display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) + "_iter"+str(i)+".png"), blur=False)
                        
            img = next_img
import sys
sys.path.append('./model')
import dino # model

import object_discovery as tokencut
import argparse
import utils
import bilateral_solver
import os

from shutil import copyfile
import PIL.Image as Image
import cv2
import numpy as np
from tqdm import tqdm

import torchvision
from torchvision import transforms
import metric
import matplotlib.pyplot as plt
import skimage
import torch

# Image transformation applied to all images
ToTensor = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225)),])


import random
import colorsys
from skimage.measure import find_contours
from matplotlib.patches import Polygon

import torch.nn.functional as F

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

def get_tokencut_binary_map(input_images, backbone, patch_size, tau) :
    # I = Image.open(img_pth).convert('RGB')
    # I_resize, w, h, feat_w, feat_h = utils.resize_pil(I, patch_size)

    # tensor = ToTensor(I_resize).unsqueeze(0).cuda()
    # feat = backbone(tensor)[0]
    bs, h, w = input_images.shape[0], input_images.shape[-2], input_images.shape[-1]
    feat_h = h // patch_size
    feat_w = w // patch_size
    outputs, feat = backbone(input_images)

    # feat = feat[0]

    seed, bipartition, eigvec = tokencut.ncut(feat, [bs, feat_h, feat_w], [patch_size, patch_size], [h,w], tau)
    return bipartition, eigvec

def mask_color_compose(org, mask, mask_color = [173, 216, 230]) :

    mask_fg = mask > 0.5
    rgb = np.copy(org)
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)

    return Image.fromarray(rgb)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## input / output dir
parser.add_argument('--out-dir', type=str, default='saliency_results', help='output directory')

parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')

parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')

parser.add_argument('--patch-size', type=int, default=8, choices=[16, 8], help='patch size')

parser.add_argument('--tau', type=float, default=0.2, help='Tau for tresholding graph')

parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')

parser.add_argument('--sigma-luma', type=float, default=16, help='sigma luma in the bilateral solver')

parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')


parser.add_argument('--dataset', type=str, default=None, choices=['ECSSD', 'DUTS', 'DUT', None], help='which dataset?')

parser.add_argument('--nb-vis', type=int, default=100, choices=[1, 200], help='nb of visualization')

parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

args = parser.parse_args()
print (args)

## feature net

if args.vit_arch == 'base' and args.patch_size == 16:
    url = "/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'small' and args.patch_size == 16:
    url = "/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    feat_dim = 384
elif args.vit_arch == 'small' and args.patch_size == 8:
    url = "/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    feat_dim = 384

backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
#    resume_path = './model/dino_vitbase16_pretrain.pth' if args.patch_size == 16 else './model/dino_vitbase8_pretrain.pth'

#    feat_dim = 768
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
#
#else :
#    resume_path = './model/dino_deitsmall16_pretrain.pth' if args.patch_size == 16 else './model/dino_deitsmall8_pretrain.pth'
#    feat_dim = 384
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)


msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
print (msg)
backbone.eval()
backbone.cuda()

if args.dataset == 'ECSSD' :
    args.img_dir = '../datasets/ECSSD/img'
    args.gt_dir = '../datasets/ECSSD/gt'

elif args.dataset == 'DUTS' :
    args.img_dir = '../datasets/DUTS_Test/img'
    args.gt_dir = '../datasets/DUTS_Test/gt'

elif args.dataset == 'DUT' :
    args.img_dir = '../datasets/DUT_OMRON/img'
    args.gt_dir = '../datasets/DUT_OMRON/gt'

elif args.dataset is None :
    args.gt_dir = None


print(args.dataset)

if args.out_dir is not None and not os.path.exists(args.out_dir) :
    os.mkdir(args.out_dir)

if args.img_path is not None:
    args.nb_vis = 1
    img_list = [args.img_path]
else:
    img_list = sorted(os.listdir(args.img_dir))

count_vis = 0
mask_lost = []
mask_bfs = []
gt = []

n_iter = 10
image_size = (480,480)
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

for img_name in tqdm(img_list) :    
    if args.img_path is not None:
        img_pth = img_name
        img_name = img_name.split("/")[-1]
        print(img_name)
    else:
        img_pth = os.path.join(args.img_dir, img_name)
        
    img = Image.open(args.img_path).convert('RGB')
    
    ref_img = transforms.Resize(image_size)(img)
    ref_img = torch.from_numpy(np.array(ref_img)).float().cuda()
    ref_img = ref_img.permute(2,0,1).unsqueeze(0)
        
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img).cuda()
    img = img.unsqueeze(0)
    orig_img = img.clone()
        
    bs = img.shape[0]
    remains = torch.ones((bs, img.shape[-2],img.shape[-1])).to(img.device)
    
    for i in range(n_iter):
        bipartition, eigvec = get_tokencut_binary_map(img, backbone, args.patch_size, args.tau)
        mask_lost.append(bipartition)

        bipartition = bipartition * remains

        output_solver, binary_solver = bilateral_solver.bilateral_solver_output(ref_img, bipartition, sigma_spatial = args.sigma_spatial, sigma_luma = args.sigma_luma, sigma_chroma = args.sigma_chroma)
               
        mask1 = bipartition
        mask2 = binary_solver
        
        mask1 = F.interpolate(mask1.unsqueeze(1), size=(60,60), mode='nearest').squeeze(1)
        mask2 = F.interpolate((mask2*1.).unsqueeze(1), size=(60,60), mode='nearest').squeeze(1)
        
        output_solver = F.interpolate(output_solver.unsqueeze(1), size=(60,60), mode='bilinear').squeeze(1)
        
        if metric.IoU(mask1, mask2) < 0.5:
            # binary_solver = binary_solver * -1
            temp = binary_solver
            binary_solver = bipartition
        mask_bfs.append(output_solver)
        
        if i == n_iter - 1:
            binary_solver = remains
        else:
            remains = (remains * (1-binary_solver*1))

        print(f'args.out_dir: {args.out_dir}, img_name: {img_name}')
        out_name = os.path.join(args.out_dir, img_name.replace('.jpg', '_iter{}.jpg'.format(i)))
        out_lost = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_iter{}.jpg'.format(i)))
        out_bfs = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_bfs_iter{}.jpg'.format(i)))
        if metric.IoU(mask1, mask2) < 0.5:
            out_temp = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_temp_iter{}.jpg'.format(i)))
        
        # org = np.array(transforms.ToPILImage()(unorm(img).squeeze(0)))

        # mask_color_compose(org, bipartition).save(out_lost)
        # mask_color_compose(org, binary_solver).save(out_bfs)

        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), out_name)
        
        image = skimage.io.imread(out_name)
        display_instances(image, bipartition.squeeze(0).cpu().numpy(), fname=out_lost, blur=False)
        display_instances(image, binary_solver.squeeze(0).cpu().numpy(), fname=out_bfs, blur=False)
        if metric.IoU(mask1, mask2) < 0.5:
            display_instances(image, temp.squeeze(0).cpu().numpy(), fname=out_temp, blur=False)   
        
        img = (img * remains).float()
        # plt.imsave(fname='{}/img_after_iter{}.png'.format(args.out_dir,i), arr=np.array(transforms.ToPILImage()(unorm(img).squeeze(0))), format='png')
        # torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), '{}/img_after_iter{}.png'.format(args.out_dir,i))
        

if args.gt_dir is not None and args.img_path is None:
    print ('TokenCut evaluation:')
    print (metric.metrics(mask_lost, gt))
    print ('\n')

    print ('TokenCut + bilateral solver evaluation:')
    print (metric.metrics(mask_bfs, gt))
    print ('\n')
    print ('\n')
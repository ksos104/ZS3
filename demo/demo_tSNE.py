# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask_former import add_mask_former_config
from predictor import VisualizationDemo

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


# constants
WINDOW_NAME = "MaskFormer demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # args = get_parser().parse_args()
    
    import easydict
    
    args = easydict.EasyDict({
        "config_file": "./configs/pascal_voc/zegformer_R101_bs32_30k_vit16_voc_gzss_eval.yaml",
        "input": ["/mnt/server14_hard1/msson/datasets/zs3_datasets/VOCZERO/images/val/*.jpg"],
        "opts": ["MODEL.WEIGHTS","./trained/given/zegformer_R101_bs32_10k_vit16_voc.pth"],
        "output": None
    })
    
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            x_cls, image_features, text_features = demo.get_features(img)
            x_cls = x_cls.squeeze()
            
            print(((x_cls @ text_features.clone().detach().t())*100))
            
            import torch
            
            ## Masks & Image & Text
            features = torch.cat((x_cls, image_features, text_features), dim=0)
            tsne_np = TSNE(n_components=2).fit_transform(features.cpu())
            tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
            
            tsne_df_x = tsne_df[:100]
            tsne_df_img = tsne_df[100:200]
            tsne_df_text = tsne_df[200:]
            
            plt.scatter(tsne_df_x['component 0'], tsne_df_x['component 1'], color = 'yellow', label = 'mask')
            plt.scatter(tsne_df_img['component 0'], tsne_df_img['component 1'], color = 'blue', label = 'image')
            plt.scatter(tsne_df_text['component 0'], tsne_df_text['component 1'], color = 'purple', label = 'text')
            
            plt.xlabel('component 0')
            plt.ylabel('component 1')
            plt.legend()
            plt.show()
            
            # ## Image & Text
            # features = torch.cat((image_features, text_features), dim=0)
            # tsne_np = TSNE(n_components=2).fit_transform(features.cpu())
            # tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
            
            # tsne_df_img = tsne_df[:100]
            # tsne_df_text = tsne_df[100:]
            
            # plt.scatter(tsne_df_img['component 0'], tsne_df_img['component 1'], color = 'blue', label = 'image')
            # plt.scatter(tsne_df_text['component 0'], tsne_df_text['component 1'], color = 'purple', label = 'text')
            
            # plt.xlabel('component 0')
            # plt.ylabel('component 1')
            # plt.legend()
            # plt.show()
            
            # ## Masks only
            # tsne_np = TSNE(n_components=2).fit_transform(x_cls.cpu())
            # tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
            
            # plt.scatter(tsne_df['component 0'], tsne_df['component 1'], color = 'blue', label = 'mask')
            
            # plt.xlabel('component 0')
            # plt.ylabel('component 1')
            # plt.legend()
            # plt.show()
            
            # ## Image only
            # tsne_np = TSNE(n_components=2).fit_transform(image_features.cpu())
            # tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
            
            # plt.scatter(tsne_df['component 0'], tsne_df['component 1'], color = 'yellow', label = 'image')
            
            # plt.xlabel('component 0')
            # plt.ylabel('component 1')
            # plt.legend()
            # plt.show()
            
            # ## Text only
            # tsne_np = TSNE(n_components=2).fit_transform(text_features.cpu())
            # tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
            
            # plt.scatter(tsne_df['component 0'], tsne_df['component 1'], color = 'purple', label = 'text')
            
            # plt.xlabel('component 0')
            # plt.ylabel('component 1')
            # plt.legend()
            # plt.show()
            
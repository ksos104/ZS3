# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, transforms
from dino_clip.third_party import imagenet_templates

import neptune.new as neptune

import os
from . import vision_transformer as vits
from dino_clip.third_party import clip
from dino_clip.modeling.heads.dino_decoder import DINODecoder
import numpy as np



@META_ARCH_REGISTRY.register()
class DINO_CLIP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        # backbone: Backbone,
        # sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        metadata_val_all,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        gzero_calibrate: float,
        clip_classification: bool,
        ensembling: bool,
        ensembling_all_cls: bool,
        train_class_json: str,
        test_class_json: str,
        clip_cls_style: str,
        ## DINO
        arch: str,
        patch_size: int,
        pretrained_weights: str,
        checkpoint_key: str,
        image_size: Tuple[int],
        threshold: float,
        n_iter: int,
        ## CLIP
        # in_channels,
        mask_classification=True,
        num_classes: int,
        hidden_dim: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
        train_class_indexes_json: str,
        test_class_indexes_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        wordvec: bool,
        temperature: float,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()

        ## MaskFormer & CLIP
        # self.backbone = backbone
        # import ipdb; ipdb.set_trace()
        # self.sem_seg_head = sem_seg_head

        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.panoptic_on = panoptic_on
        self.clip_classification = clip_classification
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        self.metadata_val_all = metadata_val_all
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        if self.clip_classification:
            self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        self.gzero_calibrate = gzero_calibrate
        self.ensembling = ensembling
        self.ensembling_all_cls = ensembling_all_cls
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        self.clip_cls_style = clip_cls_style
        assert clip_cls_style in ["crop", "mask", "cropmask"]

        if hasattr(self.metadata, "val_extra_classes"):
            val_extra_classes = self.metadata.val_extra_classes
        else:
            val_extra_classes = []
        seen_indexes = []
        unseen_indexes = []
        for cls in self.metadata.stuff_classes:
            if cls not in val_extra_classes:
                seen_indexes.append(self.metadata.stuff_classes.index(cls))
            else:
                unseen_indexes.append(self.metadata.stuff_classes.index(cls))
        self.seen_indexes = seen_indexes
        self.unseen_indexes = unseen_indexes
        
        ## DINO init
        self.patch_size = patch_size
        self.image_size = image_size
        self.threshold = threshold
        self.n_iter = n_iter
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # build model
        self.model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(device)
        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        else:
            print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
            url = None
            if arch == "vit_small" and patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif arch == "vit_small" and patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif arch == "vit_base" and patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif arch == "vit_base" and patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                self.model.load_state_dict(state_dict, strict=True)
            else:
                print("There is no reference weights available for this model => We use random weights.")
        
        # input_dim = (self.image_size[0]//self.patch_size) * (self.image_size[1]//self.patch_size) * 384
        input_dim = 384
        self.head = nn.Linear(input_dim, 512)
        
        # self.mask_layers = nn.Sequential(
        #     nn.Conv2d(input_dim, input_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(input_dim//2, input_dim//4, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(input_dim//4, 1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU()
        # )
        self.mask_layers = nn.Linear(input_dim, 1)
        
        nh = 6
        # self.attn_layers = nn.Sequential(
        #     nn.Conv2d(384+nh, (384+nh)//2, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d((384+nh)//2, (384+nh)//4, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d((384+nh)//4, nh, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU()
        # )
        
        ## DINO decoder
        self.dino_decoder = DINODecoder(
            norm = "GN",
        )
        
        
        ## CLIP init
        self.num_classes = num_classes
        self.mask_classification = mask_classification
        import json
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False)
        
        import math
        self.bg_feature = nn.Parameter(torch.Tensor(1, 512))
        nn.init.kaiming_uniform_(
            self.bg_feature, a=math.sqrt(5))
        self.bg_feature.requires_grad = True
        self.prompt_ensemble_type = prompt_ensemble_type
        # self.projection_layer = nn.Linear(hidden_dim, 512)
        
        with torch.no_grad():
            assert "A photo of" not in self.class_texts[0]
            if self.prompt_ensemble_type == "imagenet_select":
                prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
            elif self.prompt_ensemble_type == "imagenet":
                prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
            elif self.prompt_ensemble_type == "single":
                prompt_templates = ['A photo of a {} in the scene',]
            else:
                raise NotImplementedError
            prompt_templates_clip = imagenet_templates.IMAGENET_TEMPLATES_SELECT_CLIP
            
            clip_features = self.zeroshot_classifier(self.class_texts, prompt_templates, clip_model).permute(1, 0).float()
            clip_sim = clip_features.clone().detach() @ clip_features.clone().detach().t()
                                
            self.text_features = self.zeroshot_classifier(self.class_texts, prompt_templates, clip_model).permute(1, 0).float()
            self.text_features_test = self.zeroshot_classifier(self.test_class_texts, prompt_templates, clip_model).permute(1, 0).float()

            self.text_features_clip = self.zeroshot_classifier(self.class_texts, prompt_templates_clip, clip_model).permute(1, 0).float()
            self.text_features_test_clip = self.zeroshot_classifier(self.test_class_texts, prompt_templates_clip, clip_model).permute(1, 0).float()
        
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1/temperature)]).float())
        self.logit_scale.requires_grad = False
        self.clip_classification = clip_classification
        if self.clip_classification:
            self.clip_model = clip_model
            self.clip_preprocess = clip_preprocess        
        
        ## Neptune init
        self.use_npt = True
        self.rank = 0
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        if self.use_npt:
            if self.rank==0:
                self.npt = neptune.init_run(
                    project="kaist-cilab/ZS3",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4OTQ2MGY0Yi0zMTM2LTQ5ZmEtYjlmOS1lNmQxMTliOTE0MjkifQ==",
                )

    @classmethod
    def zeroshot_classifier(self, classnames, templates, clip_modelp):
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

    @classmethod
    def from_config(cls, cfg):
        # backbone = build_backbone(cfg)
        # sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        num_classes = 15

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # building criterion
        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            # sem_seg_head.num_classes,
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        metadata_val_all = MetadataCatalog.get(cfg.DATASETS.VAL_ALL[0])

        ## DINO config
        arch = cfg.DINO.ARCH
        pretrained_weights = cfg.DINO.PRE_WEIGHTS
        checkpoint_key = cfg.DINO.CKPT_KEY
        patch_size = cfg.DINO.PATCH_SIZE
        image_size = cfg.DINO.IMAGE_SIZE
        threshold = cfg.DINO.THRESHOLD
        n_iter = cfg.DINO.NUM_ITER

        ret = {
            # "backbone": backbone,
            # "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": metadata,
            "metadata_val_all": metadata_val_all,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "gzero_calibrate": cfg.MODEL.MASK_FORMER.GZERO_CALIBRATE,
            "clip_classification": cfg.MODEL.SEM_SEG_HEAD.CLIP_CLASSIFICATION,
            "ensembling": cfg.MODEL.MASK_FORMER.ENSEMBLING,
            "ensembling_all_cls": cfg.MODEL.MASK_FORMER.ENSEMBLING_ALL_CLS,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "clip_cls_style": cfg.MODEL.CLIP_CLS_STYLE,
            ## DINO
            "arch": arch,
            "pretrained_weights": pretrained_weights,
            "checkpoint_key": checkpoint_key,
            "patch_size": patch_size,
            "image_size": image_size,
            "threshold": threshold,
            "n_iter": n_iter,
        }
        ## CLIP
        # ret["in_channels"] = in_channels
        ret["mask_classification"] = True
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["train_class_indexes_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES
        ret["test_class_indexes_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE
        ret["wordvec"] = cfg.MODEL.SEM_SEG_HEAD.WORDVEC
        ret["temperature"] = cfg.MODEL.SEM_SEG_HEAD.TEMPERATURE

        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, tsne=False, mask_vis=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.clip_classification:
            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # note, after from_tensors, the images are padded, so the shape of images and batched_inputs[0]["image"] are different
        # TODO: check the add_mask operation
        # add_mask = True
        add_mask = False
        if add_mask:
            if self.training:
                ori_sizes = [x["ori_size"] for x in batched_inputs]
            else:
                ori_sizes = [(x.shape[1], x.shape[2]) for x in images]
        else:
            ori_sizes = None

        images = ImageList.from_tensors(images, self.size_divisibility)
        
        
        
        input_images = F.interpolate(images.tensor, self.image_size)
        w, h = input_images.shape[2] - input_images.shape[2] % self.patch_size, input_images.shape[3] - input_images.shape[3] % self.patch_size
        input_images = input_images[:, :, :w, :h]

        w_featmap = input_images.shape[-2] // self.patch_size
        h_featmap = input_images.shape[-1] // self.patch_size
        
        th_attn_list = []
        features_list = []
        mask_output_list = []
        attentions_list = []
        threshold = self.threshold
        
        cls_mode = 'patch-wise'      ## 'patch-wise' / 'mask-wise'
        
        if cls_mode == 'mask-wise':
            '''
                mask-wise classification
                features.shape = []
            '''
            cls_token_list = []
            if threshold is not None:
                bs = images.tensor.shape[0]
                remains = torch.ones((bs,1,3600)).to(self.device)
                th_attn = torch.zeros((bs,1,3600)).to(self.device)
                
            for i in range(self.n_iter):
                cls_tokens, features, attentions = self.model.get_attn_and_cls(input_images)
                features = features[:,1:,:]
                features_list.append(features)
                cls_token_list.append(cls_tokens)                

                bs = attentions.shape[0]
                nh = attentions.shape[1] # number of head
                
                # we keep only the output patch attention
                attentions = attentions[:, :, 0, 1:].reshape(bs, nh, -1)
                
                attentions = attentions.sum(1).unsqueeze(1)
                # attentions = attentions[:,0,:].unsqueeze(1)
                attentions_list.append(attentions)
                nh = 1

                if threshold is not None:
                    if i == self.n_iter - 1:
                        remains -= th_attn
                        th_attn = remains
                        assert remains.min() == 0
                    else:
                        remains -= th_attn
                        # we keep only a certain percentage of the mass
                        val, idx = torch.sort(attentions, dim=2)
                        val /= torch.sum(val, dim=2, keepdim=True)
                        cumval = torch.cumsum(val, dim=2)
                        th_attn = cumval > (1 - threshold)
                        idx2 = torch.argsort(idx, dim=2)
                        # for head in range(nh):
                            # th_attn[:,head] = th_attn[:,head][idx2[head]]
                        th_attn = torch.gather(th_attn, 2, idx2).float()
                        # Eliminate elements chosen before
                        th_attn = th_attn * remains
                        
                        # th_attn = th_attn.reshape(bs, nh, w_featmap, h_featmap).float()
                        # interpolate
                        # th_attn = nn.functional.interpolate(th_attn, scale_factor=self.patch_size, mode="nearest")

                # attentions = attentions.reshape(bs, nh, w_featmap, h_featmap)
                # attentions = nn.functional.interpolate(attentions, scale_factor=self.patch_size, mode="nearest")

                th_attn_list.append(th_attn)
                th_attn_imgs = th_attn.reshape(bs, nh, w_featmap, h_featmap)
                th_attn_imgs = nn.functional.interpolate(th_attn_imgs, scale_factor=self.patch_size, mode="nearest")
                
                input_images = input_images * (1-th_attn_imgs)
                
                # masked_features = (th_attn * features.permute(0,2,1)).permute(0,2,1)
                masked_features = ((th_attn*attentions) + features.permute(0,2,1)).permute(0,2,1)
                # masked_features = masked_features.reshape(bs, -1)
                mask_output = self.head(masked_features).unsqueeze(dim=1)
                mask_output_list.append(mask_output)
            
            th_attns = torch.concat(th_attn_list, dim=1).unsqueeze(dim=-1)
            features_tensor = torch.stack(features_list, dim=1)
            mask_outputs = torch.concat(mask_output_list, dim=1)
            # attentions = torch.concat(attentions_list, dim=1).reshape(bs, -1, w_featmap, h_featmap)
            # attentions = nn.functional.interpolate(attentions, size=(128,128))
            attentions = torch.concat(attentions_list, dim=1).unsqueeze(dim=-1)
            cls_tokens_tensor = torch.stack(cls_token_list, dim=1)
            
            ## Save th_attns as text
            if not self.training:
                temp = th_attns.squeeze()
                temp = temp.view(self.n_iter, 60, 60)
                for i in range(self.n_iter):
                    np.savetxt('temp{}.txt'.format(i), temp[i,...].cpu().numpy().astype(int), fmt='%d')
            
            ## ZegOT: Global Text Alignment
            # cls_tokens_tensor = mask_outputs
            # align_factor = self.align_layer(torch.cat(((cls_tokens_tensor*self.text_features), self.text_features)))
            
            
            if self.mask_classification:
                mask_outputs = mask_outputs / mask_outputs.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                if self.training:
                    cls_score = logit_scale * mask_outputs @ self.text_features.clone().detach().t()
                else:
                    cls_score = logit_scale * mask_outputs @ self.text_features_test.clone().detach().t()

                bg_score = logit_scale * mask_outputs @ self.bg_feature.t()
                outputs_class = torch.cat((cls_score, bg_score), -1)
                outputs = {"pred_logits": outputs_class}
            else:
                outputs = {}
                
            outputs['semantic_vector'] = mask_outputs
            
            ## outputs['pred_masks].shape = [bs, 5, 60, 60]
            # pred_masks = th_attns * attentions
            # mask_inputs = (features_tensor * (th_attns * attentions)).permute(0,1,3,2).reshape(bs * self.n_iter, -1, w_featmap, h_featmap)        
            # pred_masks = self.mask_layers(mask_inputs).reshape(bs, self.n_iter, -1, w_featmap, h_featmap).squeeze(dim=2)
            
            # mask_inputs = (features_tensor * (th_attns * attentions)).contiguous().view(bs, -1, 384)
            mask_inputs = (features_tensor * attentions).contiguous().view(bs, -1, 384)
            mask_inputs = self.mask_layers(mask_inputs)
            # pred_masks = self.mask_layers(mask_inputs).reshape(bs, self.n_iter, -1, w_featmap, h_featmap).squeeze(dim=2)
            
            pred_masks = mask_inputs.contiguous().view(bs, self.n_iter, w_featmap, h_featmap)
            outputs['pred_masks'] = pred_masks

            if self.training:
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                # bipartite matching-based loss
                losses = self.criterion(outputs, targets)
                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)

                if self.use_npt and self.rank==0:    
                    self.npt["train/loss_ce"].append(losses["loss_ce"])
                    self.npt["train/loss_mask"].append(losses["loss_mask"])
                    self.npt["train/loss_dice"].append(losses["loss_dice"])
                    self.npt["train/loss_total"].append(sum(losses.values()))

                return losses
            ##################
            ##  Evaluation  ##
            ##################
            else:
                return
        elif cls_mode == 'patch-wise':
            '''
                patch-wise classification
                features.shape = [bs, 3600, 384]
            '''
            if threshold is not None:
                bs = images.tensor.shape[0]
                remains = torch.ones((bs,1,3600)).to(self.device)
                th_attn = torch.zeros((bs,6,3600)).to(self.device)
                th_attn_sum = torch.zeros((bs,1,3600)).to(self.device)
            for i in range(self.n_iter):
                features, attentions = self.model.get_last_selfattention(input_images)
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
                features, attentions = self.dino_decoder(features, attentions)
                nh = 1
                
                '''
                    Watershed algorithm
                '''
                # ## Select a mask has the highest attention value
                # from skimage.segmentation import watershed
                # from skimage.feature import peak_local_max
                # from scipy import ndimage as ndi

                # def find_consecutive_mask(heatmap):
                #     heatmap = heatmap.reshape(heatmap.shape[0], heatmap.shape[1], 60, 60)
                    
                #     heatmap = heatmap[0,0,...]
                    
                #     # Find the local maxima in the heatmap
                #     # local_maxima = peak_local_max(heatmap.cpu().numpy(), threshold_abs=torch.mean(heatmap.cpu()).numpy(), exclude_border=False, min_distance=60)
                #     local_maxima = peak_local_max(heatmap.cpu().numpy(), exclude_border=False, min_distance=60)
                    
                #     mask = np.zeros(heatmap.shape, dtype=bool)
                #     mask[tuple(local_maxima.T)] = True
                #     markers, _ = ndi.label(mask)
                    
                #     # Perform watershed segmentation on the heatmap
                #     labels = watershed(-heatmap.cpu().numpy(), markers=markers, mask=(heatmap >= torch.mean(heatmap)).cpu())
                #     labels = torch.from_numpy(labels).to(self.device)
                    
                #     # Find the label corresponding to the highest peak
                #     max_label = labels[np.unravel_index(torch.argmax(heatmap).cpu(), heatmap.shape)]
                    
                #     # Create a mask that covers the highest peak and its surrounding hill over the threshold
                #     mask = torch.zeros_like(heatmap).cuda()
                #     mask[labels == max_label] = 1
                #     mask[heatmap < torch.mean(heatmap)] = 0

                #     # Convert mask back to a numpy array and return
                #     # return mask.numpy()
                #     return mask * heatmap

                # ref_attentions = find_consecutive_mask(attentions)
            
                '''
                    refine attentions through FCN
                '''
                # attentions = torch.cat((features.permute(0,2,1), attentions), dim=1)
                # attentions_img = attentions.reshape(bs, features.shape[-1]+nh, 60, 60)
                # attentions_img = self.attn_layers(attentions_img)
                # attentions = attentions_img.reshape(bs, nh, -1)                
                
                features_list.append(features)
                attentions_list.append(attentions)

                if i == self.n_iter - 1:
                    remains -= th_attn_sum
                    th_attn = remains.repeat([1,nh,1])
                    assert remains.min() >= 0
                else:
                    remains -= th_attn_sum
                    # we keep only a certain percentage of the mass
                    val, idx = torch.sort(attentions, dim=2)
                    val /= torch.sum(val, dim=2, keepdim=True)
                    cumval = torch.cumsum(val, dim=2)
                    th_attn = cumval > (1 - threshold)
                    idx2 = torch.argsort(idx, dim=2)
                    # for head in range(nh):
                        # th_attn[:,head] = th_attn[:,head][idx2[head]]
                    th_attn = torch.gather(th_attn, 2, idx2).float()
                    # Eliminate elements chosen before
                    th_attn = th_attn * remains
                th_attn_sum = th_attn.sum(dim=1).unsqueeze(dim=1)

                th_attn_list.append(th_attn)
                
                th_attn_sum = torch.where(th_attn_sum==0, th_attn_sum, torch.tensor([1.],device=self.device))
                th_attn_imgs = th_attn_sum.reshape(bs, 1, w_featmap, h_featmap)
                th_attn_imgs = nn.functional.interpolate(th_attn_imgs, scale_factor=self.patch_size, mode="nearest")
                
                input_images = input_images * (1-th_attn_imgs)
                
                masked_features = ((th_attn * attentions).unsqueeze(dim=-1) + features.unsqueeze(dim=1).repeat([1,nh,1,1]))
                mask_output = self.head(masked_features)
                mask_output_list.append(mask_output)
            
            th_attns = torch.concat(th_attn_list, dim=1).unsqueeze(dim=-1)
            features_tensor = torch.stack(features_list, dim=1)
            mask_outputs = torch.concat(mask_output_list, dim=1)
            attentions = torch.concat(attentions_list, dim=1).unsqueeze(dim=-1)
            
            ## Attentions refinement
            bin_attns = th_attns * attentions
            bin_attns = bin_attns.reshape(bs, nh, self.n_iter, -1, 1)
            bin_attns = (bin_attns == bin_attns.max(dim=1, keepdim=True)[0]).to(bin_attns)
            bin_attns = bin_attns.reshape(bs, nh*self.n_iter, -1, 1)
            attentions = bin_attns * attentions
            
            ## Save th_attns as text
            # if not self.training:
            #     temp = th_attns.squeeze()
            #     temp = temp.view(self.n_iter, 60, 60)
            #     for i in range(self.n_iter):
            #         np.savetxt('temp{}.txt'.format(i), temp[i,...].cpu().numpy().astype(int), fmt='%d')
            
            ## ZegOT: Global Text Alignment
            # cls_tokens_tensor = mask_outputs
            # align_factor = self.align_layer(torch.cat(((cls_tokens_tensor*self.text_features), self.text_features)))
            
            
            if self.mask_classification:
                mask_outputs = mask_outputs / mask_outputs.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                if self.training:
                    cls_score = logit_scale * mask_outputs @ self.text_features.clone().detach().t()
                else:
                    cls_score = logit_scale * mask_outputs @ self.text_features_test.clone().detach().t()

                bg_score = logit_scale * mask_outputs @ self.bg_feature.t()
                outputs_class = torch.cat((cls_score, bg_score), -1)
                outputs = {"pred_logits": outputs_class}
            else:
                outputs = {}
            outputs['semantic_vector'] = mask_outputs
                        
            mask_inputs = (features_tensor.repeat([1,nh,1,1]) + (th_attns * attentions)).contiguous().view(bs, -1, 384)
            mask_inputs = self.mask_layers(mask_inputs)
            
            pred_masks = mask_inputs.contiguous().view(bs, self.n_iter*nh, w_featmap, h_featmap)
            outputs['pred_masks'] = pred_masks


            ## Training or Inference
            if self.training:
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                # bipartite matching-based loss
                losses = self.criterion(outputs, targets)
                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)

                if self.use_npt and self.rank==0:    
                    self.npt["train/loss_ce"].append(losses["loss_ce"])
                    self.npt["train/loss_mask"].append(losses["loss_mask"])
                    self.npt["train/loss_dice"].append(losses["loss_dice"])
                    self.npt["train/loss_total"].append(sum(losses.values()))

                return losses
            else:
                mask_cls_results = outputs["pred_logits"]           ## [bs, n_heads*n_iter, 3600, n_cls+1]
                mask_pred_results = outputs["pred_masks"]           ## [bs, n_heads*n_iter, 60, 60]
                
                ## mask_pred_results.shape = [bs, n_iter*3600, 1, 1]
                # mask_pred_results = mask_pred_results.squeeze(-1).squeeze(-1)
                # mask_pred_results = mask_pred_results.view(bs, mask_pred_results.shape[-1]//3600, 60, 60)
                
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                self.clip_classification = False
                if self.clip_classification:
                    ##########################
                    mask_pred_results_224 = F.interpolate(mask_pred_results,
                        size=(224, 224), mode="bilinear", align_corners=False,)
                    images_tensor = F.interpolate(clip_images.tensor,
                                                size=(224, 224), mode='bilinear', align_corners=False,)
                    mask_pred_results_224 = mask_pred_results_224.sigmoid() > 0.5

                    mask_pred_results_224 = mask_pred_results_224.unsqueeze(2)

                    masked_image_tensors = (images_tensor.unsqueeze(1) * mask_pred_results_224)
                    cropp_masked_image = False
                    # vis_cropped_image = True
                    if cropp_masked_image:
                        # import ipdb; ipdb.set_trace()
                        mask_pred_results_ori = mask_pred_results
                        mask_pred_results_ori = mask_pred_results_ori.sigmoid() > 0.5
                        mask_pred_results_ori = mask_pred_results_ori.unsqueeze(2)
                        masked_image_tensors_ori = (clip_images.tensor.unsqueeze(1) * mask_pred_results_ori)
                        # TODO: repeat the clip_images.tensor to get the non-masked images for later crop.
                        ori_bs, ori_num_queries, ori_c, ori_h, ori_w = masked_image_tensors_ori.shape
                        # if vis_cropped_image:
                        clip_images_repeat = clip_images.tensor.unsqueeze(1).repeat(1, ori_num_queries, 1, 1, 1)
                        clip_images_repeat = clip_images_repeat.reshape(ori_bs * ori_num_queries, ori_c, ori_h, ori_w)

                        masked_image_tensors_ori = masked_image_tensors_ori.reshape(ori_bs * ori_num_queries, ori_c, ori_h, ori_w)
                        import torchvision
                        # binary_mask_preds: [1, 100, 512, 704]
                        binary_mask_preds = mask_pred_results.sigmoid() > 0.5
                        binary_bs, binary_num_queries, binary_H, binary_W = binary_mask_preds.shape
                        # assert binary_bs == 1
                        binary_mask_preds = binary_mask_preds.reshape(binary_bs * binary_num_queries,
                                                                    binary_H, binary_W)
                        sum_y = torch.sum(binary_mask_preds, dim=1)
                        cumsum_x = torch.cumsum(sum_y, dim=1).float()
                        xmaxs = torch.argmax(cumsum_x, dim=1, keepdim=True) # shape: [100, 1]
                        cumsum_x[cumsum_x==0] = np.inf
                        xmins = torch.argmin(cumsum_x, dim=1, keepdim=True)
                        sum_x = torch.sum(binary_mask_preds, dim=2)
                        cumsum_y = torch.cumsum(sum_x, dim=1).float()
                        ymaxs = torch.argmax(cumsum_y, dim=1, keepdim=True)
                        cumsum_y[cumsum_y==0] = np.inf
                        ymins = torch.argmin(cumsum_y, dim=1, keepdim=True)
                        areas = (ymaxs - ymins) * (xmaxs - xmins)
                        ymaxs[areas == 0] = images.tensor.shape[-2]
                        ymins[areas == 0] = 0
                        xmaxs[areas == 0] = images.tensor.shape[-1]
                        xmins[areas == 0] = 0
                        boxes = torch.cat((xmins, ymins, xmaxs, ymaxs), 1)  # [binary_bs * binary_num_queries, 4]
                        # boxes = boxes.reshape(binary_bs, binary_num_queries, 4)
                        # TODO: crop images by boxes in the original image size
                        # boxes_list = [boxes[i].reshape(1, -1) for i in range(boxes.shape[0])]
                        boxes_list = []
                        for i in range(boxes.shape[0]):
                            boxes_list.append(boxes[i].reshape(1, -1).float())
                        box_masked_images = torchvision.ops.roi_align(masked_image_tensors_ori, boxes_list, 224, aligned=True)
                        box_masked_images = box_masked_images.reshape(ori_bs, ori_num_queries, ori_c, 224, 224)

                        # if vis_cropped_image:
                            # import ipdb; ipdb.set_trace()
                        box_images = torchvision.ops.roi_align(clip_images_repeat, boxes_list, 224, aligned=True)
                        box_images = box_images.reshape(ori_bs, ori_num_queries, ori_c, 224, 224)

                    count = 0
                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])

                    if self.clip_classification:
                        masked_image_tensor = masked_image_tensors[count]
                        # if cropp_masked_image:
                        box_masked_image_tensor = box_masked_images[count]
                        # if vis_cropped_image:
                        box_image_tensor = box_images[count]
                        # boxs = boxes_list[count]
                        count = count + 1

                        with torch.no_grad():
                            if self.clip_cls_style == "cropmask":
                                clip_input_images = box_masked_image_tensor
                            elif self.clip_cls_style == "mask":
                                clip_input_images = masked_image_tensor
                            elif self.clip_cls_style == "crop":
                                clip_input_images = box_image_tensor
                            else:
                                raise NotImplementedError

                            image_features = self.clip_model.encode_image(clip_input_images)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            logit_scale = self.clip_model.logit_scale.exp()
                            logits_per_image = logit_scale.half() * image_features @ self.text_features_test_clip.t().half()
                            # logits_per_image = logit_scale.half() * image_features @ self.sem_seg_head.predictor.text_features_test_clip.half()       ## for CoCoOp
                            logits_per_image = logits_per_image.squeeze(0)
                            logits_per_image = logits_per_image.float()
                            
                            ## mask_pred_results.shape = [bs, n_iter*3600, 1, 1]
                            logits_per_image = logits_per_image.unsqueeze(1)
                            logits_per_image = logits_per_image.repeat(1,3600,1).view(-1,self.num_classes)
                            
                            mask_cls_result = mask_cls_result.reshape(-1, mask_cls_result.shape[-1])
                            
                            logits_per_image = torch.cat((logits_per_image, mask_cls_result[:, -1].unsqueeze(1)), 1)
                            
                            if mask_vis:
                                return mask_pred_results, cls_score, logits_per_image
                            
                            assert not (self.ensembling and self.ensembling_all_cls)
                            if self.ensembling:
                                # note that in this branch, average the seen score of clip
                                # seen_indexes, unseen_indexes = self.seen_unseen_indexes()
                                lambda_balance = 2 / 3.
                                mask_cls_result = F.softmax(mask_cls_result, dim=-1)[..., :-1]
                                # shape of logits_per_image: [100, 171]
                                logits_per_image = F.softmax(logits_per_image, dim=-1)[..., :-1]
                                # remove the influence of clip on seen classes
                                logits_per_image[:, self.seen_indexes] = logits_per_image[:, self.seen_indexes].mean(dim=1, keepdim=True)

                                mask_cls_result[:, self.seen_indexes] = torch.pow(mask_cls_result[:, self.seen_indexes], lambda_balance) \
                                                                * torch.pow(logits_per_image[:, self.seen_indexes], 1 - lambda_balance)
                                mask_cls_result[:, self.unseen_indexes] = torch.pow(mask_cls_result[:, self.unseen_indexes], 1 - lambda_balance) \
                                                                * torch.pow(logits_per_image[:, self.unseen_indexes], lambda_balance)
                            elif self.ensembling_all_cls:
                                lambda_balance = 2 / 3.
                                mask_cls_result = F.softmax(mask_cls_result, dim=-1)[..., :-1]
                                logits_per_image = F.softmax(logits_per_image, dim=-1)[..., :-1]
                                mask_cls_result = torch.pow(mask_cls_result, 1 - lambda_balance) \
                                                                * torch.pow(logits_per_image, lambda_balance)
                            else:
                                mask_cls_result = logits_per_image

                        ######################################################################################
                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = sem_seg_postprocess(
                            mask_pred_result, image_size, height, width
                        )

                    ## mask_pred_results.shape = [bs, n_iter*3600, 1, 1]
                    if self.clip_classification:
                        mask_cls_result = mask_cls_result.view(mask_cls_result.shape[0]//3600, 3600, mask_cls_result.shape[1])
                    ## mask_cls_result: [bs, 3600, n_cls+1] --> [bs, n_cls+1]
                    mask_cls_result = mask_cls_result.mean(dim=1)
                    # mask_cls_result = mask_cls_result.max(dim=1)[0]
                    

                    # semantic segmentation inference
                    if (self.clip_classification and self.ensembling) or (self.clip_classification and self.ensembling_all_cls):
                        r = self.semantic_inference2(mask_cls_result, mask_pred_result)
                    else:
                        r = self.semantic_inference(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = sem_seg_postprocess(r, image_size, height, width)
                    #############################################################################
                    # gzero calibrate
                    if self.gzero_calibrate > 0:
                        r[self.seen_indexes, :, :] = r[self.seen_indexes, :, :] - self.gzero_calibrate
                    ###########################################################################
                    processed_results.append({"sem_seg": r})

                return processed_results

    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            # import ipdb; ipdb.set_trace()
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        return semseg

    def semantic_inference2(self, mask_cls, mask_pred):
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
    
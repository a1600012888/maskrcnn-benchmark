# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        # fix backbone:
        #for param in self.backbone.parameters():
        #    param.requires_grad = False

        self.DOMAIN_SCC = cfg.MODEL.BACKBONE.DOMAIN_SCC
        self.SCC_DIVERSE_REG = cfg.MODEL.BACKBONE.SCC_DIVERSE_REG

        if self.DOMAIN_SCC:
            if self.SCC_DIVERSE_REG:
                from tylib.tytorch.layers.diverse_reg_domain_scc_modifier import Conv2DScc
                alpha = cfg.MODEL.SCC.REG_LOSS.ALPHA
                beta = cfg.MODEL.SCC.REG_LOSS.BETA
                lamda = cfg.MODEL.SCC.REG_LOSS.LAMBDA

                Conv2DScc(self.backbone, alpha, beta, lamda,
                          cfg.MODEL.BACKBONE.NUM_EXPERTS,
                          cfg.MODEL.BACKBONE.IN_NUM)
            else:
                from tylib.tytorch.layers.domain_scc_modifier import Conv2DScc

                Conv2DScc(self.backbone, cfg.MODEL.BACKBONE.NUM_EXPERTS,
                          cfg.MODEL.BACKBONE.IN_NUM)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, embedding=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """


        #print(embedding.size(), 'em')
        #print(_, '_')
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        if self.DOMAIN_SCC:
            embedding = torch.stack(embedding, 0)
            embedding = embedding.to(images.tensors.get_device())
            #print(embedding.type())
            embedding = embedding.type(torch.float)
            self.backbone.embedding_vec = embedding
            #features = self.backbone(images.tensors, embedding)

        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        ## added by me!
        # losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)
        return result

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn
from time import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform_t = AverageMeter()
        self.backbone_t = AverageMeter()
        self.rpn_t = AverageMeter()
        self.roi_heads_t = AverageMeter()
        self.post_transform_t = AverageMeter()

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        t = time()
        images, targets = self.transform(images, targets)

        flag_t = False

        if flag_t:
            torch.cuda.synchronize()
            self.transform_t.update(time() - t)
            t = time()

        features = self.backbone(images.tensors)

        if flag_t:
            torch.cuda.synchronize()
            self.backbone_t.update(time() - t)
            t = time()

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

        if flag_t:
            torch.cuda.synchronize()
            self.rpn_t.update(time() - t)
            t = time()

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        if flag_t:
            torch.cuda.synchronize()
            self.roi_heads_t.update(time() - t)
            t = time()

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if flag_t:
            torch.cuda.synchronize()
            self.post_transform_t.update(time() - t)

            print("       {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                self.transform_t.avg,
                self.backbone_t.avg,
                self.rpn_t.avg,
                self.roi_heads_t.avg,
                self.post_transform_t.avg))

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:55:29 2022

@author: Modified by Cheng Yuxuan
"""
import torch.nn as nn

from .yolo_head_rewrite import YOLOXHead
from .yolo_pafpn_rewrite import YOLOPAFPN


class YOLOX(nn.Module):

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, ratio_scale, loss_scale = self.head(fpn_outs, targets, x)
            outputs = {"total_loss" : loss,
                       "iou_loss"   : iou_loss,
                       "l1_loss"    : l1_loss,
                       "conf_loss"  : conf_loss,
                       "cls_loss"   : cls_loss,
                       "num_fg"     : num_fg,
                       "ratio_scale": ratio_scale,
                       "loss_scale" : loss_scale}
        else:
            outputs = self.head(fpn_outs)
        return outputs
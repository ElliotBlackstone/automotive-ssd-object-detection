import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image
from torchvision.ops import box_convert
from torchvision.ops import box_iou

import pathlib
from PIL import Image
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sklearn.model_selection import train_test_split
from collections import OrderedDict



class mySSD(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super(mySSD, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes


# image size must be 300x300
#################### begin VGG16 model ####################

# BatchNorm2d was implemented after every convolution layer
# BatchNorm2d was not around when VGG was created
        self.VGG16_UpTo_conv4_3 = nn.Sequential(OrderedDict([
            ("conv1", nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels,
                                              out_channels=64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, in_channels, 300, 300) -> (B, 64, 300, 300)
                                    nn.BatchNorm2d(num_features=64),       # no size change
                                    nn.ReLU(inplace=True),                 # no size change
                                    nn.Conv2d(in_channels=64,
                                              out_channels=64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, 64, 300, 300) -> (B, 64, 300, 300)
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(inplace=True)
                                )),
            ("mp1", nn.MaxPool2d(kernel_size=2, stride=2)),                # (B, 64, 300, 300) -> (B, 64, 150, 150)
            ("conv2", nn.Sequential(
                                    nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, 64, 150, 150) -> (B, 128, 150, 150)
                                    nn.BatchNorm2d(num_features=128), 
                                    nn.ReLU(inplace=True),                        
                                    nn.Conv2d(in_channels=128,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, 128, 150, 150) -> (B, 128, 150, 150)
                                    nn.BatchNorm2d(num_features=128), 
                                    nn.ReLU(inplace=True)                         
                                    )),
            ("mp2", nn.MaxPool2d(kernel_size=2, stride=2)),                # (B, 128, 150, 150) -> (B, 128, 75, 75)
            ("conv3", nn.Sequential(
                                    nn.Conv2d(in_channels=128,
                                              out_channels=256,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, 128, 75, 75) -> (B, 256, 75, 75)
                                    nn.BatchNorm2d(num_features=256), 
                                    nn.ReLU(inplace=True),                        
                                    nn.Conv2d(in_channels=256,
                                              out_channels=256,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, 256, 75, 75) -> (B, 256, 75, 75)
                                    nn.BatchNorm2d(num_features=256), 
                                    nn.ReLU(),                        
                                    nn.Conv2d(in_channels=256,
                                              out_channels=256,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, 256, 75, 75) -> (B, 256, 75, 75)
                                    nn.BatchNorm2d(num_features=256), 
                                    nn.ReLU(inplace=True)                         
                                    )),
            ("mp3", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)),# (B, 256, 75, 75) -> (B, 256, 38, 38)   ceil_mode=True needed to round up
            ("conv4", nn.Sequential(
                                    nn.Conv2d(in_channels=256,
                                              out_channels=512,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, 256, 38, 38) -> (B, 512, 38, 38)
                                    nn.BatchNorm2d(num_features=512), 
                                    nn.ReLU(inplace=True),                        
                                    nn.Conv2d(in_channels=512,
                                              out_channels=512,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, 512, 38, 38) -> (B, 512, 38, 38)
                                    nn.BatchNorm2d(num_features=512), 
                                    nn.ReLU(inplace=True),                        
                                    nn.Conv2d(in_channels=512,
                                              out_channels=512,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),                  # (B, 512, 38, 38) -> (B, 512, 38, 38)
                                    nn.BatchNorm2d(num_features=512), 
                                    nn.ReLU(inplace=True)                         
                                    ))
        ]))

        self.VGG16_extras = nn.Sequential(OrderedDict([
            ("mp4", nn.MaxPool2d(kernel_size=2, stride=2)),                # (B, 512, 38, 38) -> (B, 512, 19, 19)
            ("conv5", nn.Sequential(
                                    nn.Conv2d(in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1),                    # (B, 512, 19, 19) -> (B, 512, 19, 19)
                                    nn.BatchNorm2d(num_features=512), 
                                    nn.ReLU(inplace=True),                        
                                    nn.Conv2d(in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1),                    # (B, 512, 19, 19) -> (B, 512, 19, 19)
                                    nn.BatchNorm2d(num_features=512), 
                                    nn.ReLU(inplace=True),                        
                                    nn.Conv2d(in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1),                    # (B, 512, 19, 19) -> (B, 512, 19, 19)
                                    nn.BatchNorm2d(num_features=512), 
                                    nn.ReLU(inplace=True)
                                    ))
        ]))

#################### end VGG16 model ####################



# Additional layers for SSD
        # PyTorch built in SSD300 has a maxpool2d layer here, and padding=stride=6 on the first conv2d layer
        self.extra_conv6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=6, dilation=6), # (B, 1024, 19, 19)
                nn.BatchNorm2d(num_features=1024),
                nn.ReLU(inplace=True)
            )

        self.extra_conv7 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1), # (B, 1024, 19, 19)
                nn.BatchNorm2d(num_features=1024),
                nn.ReLU(inplace=True)
            )
        
        self.extra_conv8_2 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1),                        # (B, 256, 19, 19)
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),              # (B, 512, 10, 10)
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(inplace=True)
            )
        
        self.extra_conv9_2 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),                         # (B, 128, 10, 10)
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),              # (B, 256, 5, 5)
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True)
            )
        
        self.extra_conv10_2 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),                         # (B, 128, 5, 5)
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),                         # (B, 256, 3, 3)
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True)
            )
        
        self.extra_conv11_2 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),                         # (B, 128, 3, 3)
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),                         # (B, 256, 1, 1)
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True)
            )

        
        # Localization and class prediction layers
        self.box_head = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),   # applied to VGG16_UpTo_conv4_3 - output size: (B, 16, 38, 38)
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),  # applied to extra_conv7        - output size: (B, 24, 19, 19)
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),   # applied to extra_conv8_2      - output size: (B, 24, 10, 10)
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),   # applied to extra_conv9_2      - output size: (B, 24, 5, 5)
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),   # applied to extra_conv10_2     - output size: (B, 16, 3, 3)
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)    # applied to extra_conv11_2     - output size: (B, 16, 1, 1)
        ])

        self.cls_head = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),  # applied to VGG16_UpTo_conv4_3 - output size: (B, 4*num_classes, 38, 38)
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1), # applied to extra_conv7        - output size: (B, 6*num_classes, 19, 19)
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),  # applied to extra_conv8_2      - output size: (B, 6*num_classes, 10, 10)
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),  # applied to extra_conv9_2      - output size: (B, 6*num_classes, 5, 5)
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),  # applied to extra_conv10_2     - output size: (B, 4*num_classes, 3, 3)
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)   # applied to extra_conv11_2     - output size: (B, 4*num_classes, 1, 1)
        ])

        # total detections per class: 4*38*38 + 6*19*19 + 6*10*10 + 6*5*5 + 4*3*3 + 4*1*1 = 8732


    def forward(self, x):

        x = self.VGG16_UpTo_conv4_3(x)
        x_VGG16_conv43 = x # will use later, no need to recompute

        x = self.VGG16_extras(x)
        x = self.extra_conv6(x)

        x_conv7 = self.extra_conv7(x)
        x_conv8 = self.extra_conv8_2(x_conv7)
        x_conv9 = self.extra_conv9_2(x_conv8)
        x_conv10 = self.extra_conv10_2(x_conv9)
        x_conv11 = self.extra_conv11_2(x_conv10)

        # apply localization head box
        loc_list = [self.box_head[0](x_VGG16_conv43).permute(0, 2, 3, 1).contiguous(),  # (B, 38, 38, 4*4)
                    self.box_head[1](x_conv7).permute(0, 2, 3, 1).contiguous(),         # (B, 19, 19, 6*4)
                    self.box_head[2](x_conv8).permute(0, 2, 3, 1).contiguous(),         # (B, 10, 10, 6*4)
                    self.box_head[3](x_conv9).permute(0, 2, 3, 1).contiguous(),         # (B, 5, 5, 6*4)
                    self.box_head[4](x_conv10).permute(0, 2, 3, 1).contiguous(),        # (B, 3, 3, 4*4)
                    self.box_head[5](x_conv11).permute(0, 2, 3, 1).contiguous()]        # (B, 1, 1, 4*4)
        
        # apply classification head box
        cls_list = [self.cls_head[0](x_VGG16_conv43).permute(0, 2, 3, 1).contiguous(),  # (B, 38, 38, 4*num_classes)
                    self.cls_head[1](x_conv7).permute(0, 2, 3, 1).contiguous(),         # (B, 19, 19, 6*num_classes)
                    self.cls_head[2](x_conv8).permute(0, 2, 3, 1).contiguous(),         # (B, 10, 10, 6*num_classes)
                    self.cls_head[3](x_conv9).permute(0, 2, 3, 1).contiguous(),         # (B, 5, 5, 6*num_classes)
                    self.cls_head[4](x_conv10).permute(0, 2, 3, 1).contiguous(),        # (B, 3, 3, 4*num_classes)
                    self.cls_head[5](x_conv11).permute(0, 2, 3, 1).contiguous()]        # (B, 1, 1, 4*num_classes)
        
        #flatten
        loc_output = torch.cat([o.view(o.size(0), -1) for o in loc_list], 1)            # (B, 34928 = 8732*4)
        cls_output = torch.cat([o.view(o.size(0), -1) for o in cls_list], 1)            # (B, 8732*num_classes) 
        
        loc_bbox_form = loc_output.view(loc_output.size(0), -1, 4)                      # (B, 8732, 4)           - box regression predictions per prior
        cls_preds = cls_output.view(cls_output.size(0), -1, self.num_classes)           # (B, 8732, num_classes) - logits per prior
                                                                                        # in this case, 8732 is the number of priors
        return loc_bbox_form, cls_preds




    def create_default_boxes(s_min=0.2, s_max=0.9, clip=True):
        """
        Create default boxes.
        Default settings create boxes as per SSD paper https://arxiv.org/abs/1512.02325

        Inputs:
        s_min - float between 0 and 1, default 0.2
        s_max - float between s_min and 1, default 0.9
        clip - bool, default True

        Output:
        Tensor of shape [8732, 4] where boxes are normalized and are of the form (cx, cy, w, h)
        """
        feature_map_sizes = [(38, 38), (19, 19), (10,10), (5,5), (3,3), (1,1)]
        aspect_ratios_per_level = [[2], [2,3], [2,3], [2,3], [2], [2]]

        L = len(feature_map_sizes)
        
        # scales s_0..s_{L-1}, and s_L = 1.0 for the s'_l computation
        s = [s_min + (s_max - s_min) * (l / (L - 1)) for l in range(L)]
        s.append(1.0)  # s_L

        priors = []
        for l, (H, W) in enumerate(feature_map_sizes):
            s_l  = s[l]
            s_lp = np.sqrt(s[l] * s[l+1])  # extra square

            # per-location widths/heights to emit, in (w,h), normalized
            whs = [(s_l, s_l), (s_lp, s_lp)]
            for a in aspect_ratios_per_level[l]:
                # add both a and 1/a unless a==1 (which we already handled)
                sr = np.sqrt(a)
                whs.append((s_l * sr, s_l / sr))
                whs.append((s_l / sr, s_l * sr))

            # tile over centers
            for i in range(H):
                cy = (i + 0.5) / H
                for j in range(W):
                    cx = (j + 0.5) / W
                    for (w, h) in whs:
                        priors.append([cx, cy, w, h])

        priors = torch.tensor(priors, dtype=torch.float32)
        if clip:
            # keep centers in [0,1], clip sizes to [eps,1]
            eps = 1e-6
            priors[:, 0:2].clamp_(0.0, 1.0)
            priors[:, 2:4].clamp_(eps, 1.0)
        return priors  # shape [num_priors, 4], (cx,cy,w,h) normalized



    def encode_ssd(gt_boxes_cxcywh, gt_labels, priors_cxcywh, iou_thresh=0.5, variances=(0.1, 0.2), background_class=0):
        """
        gt_boxes_cxcywh: [G,4] in normalized [0,1] with format (cx,cy,w,h)
        gt_labels:     [G]  integers in {1..C-1} (no background here)
        priors_cxcywh: [P,4] normalized priors (cx,cy,w,h)
        Returns:
        loc_target:  [P,4] (tx,ty,tw,th)
        cls_target:  [P]   in {background_class,..,C-1}
        pos_mask:    [P]   boolean positives
        matched_gt:  [P,4] GT boxes matched to each prior (xyxy, normalized)
        """
        P = priors_cxcywh.size(0)
        G = gt_boxes_cxcywh.size(0)
        assert G > 0, "No ground-truth boxes to encode."

        # 1) IoU matrix in xyxy space
        iou = box_iou(box_convert(boxes=priors_cxcywh, in_fmt="cxcywh", out_fmt="xyxy"),
                      box_convert(boxes=gt_boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")) # [P, G]
        # priors_xyxy = cxcywh_to_xyxy(priors_cxcywh)
        # iou = iou_xyxy(priors_xyxy, gt_boxes_xyxy)  # [P,G]

        # 2) For each GT, force-match its best prior (bipartite trick)
        best_prior_per_gt = iou.argmax(dim=0)           # [G]
        iou[best_prior_per_gt, torch.arange(G)] = 2.0   # ensure > any threshold

        # 3) For each prior, pick best GT
        best_gt_per_prior = iou.argmax(dim=1)           # [P]
        best_iou_per_prior = iou.gather(1, best_gt_per_prior.unsqueeze(1)).squeeze(1)

        pos_mask = best_iou_per_prior >= iou_thresh
        gt_cxcywh = gt_boxes_cxcywh[best_gt_per_prior]       # [P,4]
        matched_labels = gt_labels[best_gt_per_prior]        # [P]

        v_c, v_s = variances

        # offsets
        t_xy = (gt_cxcywh[:, :2] - priors_cxcywh[:, :2]) / priors_cxcywh[:, 2:] / v_c
        t_wh = torch.log(gt_cxcywh[:, 2:] / priors_cxcywh[:, 2:].clamp(min=1e-6)) / v_s
        loc_target = torch.zeros_like(priors_cxcywh)
        loc_target[:, :2] = t_xy
        loc_target[:, 2:] = t_wh

        # 5) Class targets: background for negatives
        cls_target = torch.full((P,), background_class, dtype=matched_labels.dtype, device=matched_labels.device)
        cls_target[pos_mask] = matched_labels[pos_mask]

        return loc_target, cls_target, pos_mask, gt_cxcywh



    def decode_ssd(loc, priors, variances=(0.1, 0.2)):
        """
        loc:   [num_priors, 4]  (tx, ty, tw, th)
        priors:[num_priors, 4]  (cx_a, cy_a, w_a, h_a), normalized [0,1]
        returns boxes_xyxy normalized to [0,1], shape [num_priors, 4]
        """
        v_c, v_s = variances
        # centers
        cx = loc[:, 0] * v_c * priors[:, 2] + priors[:, 0]
        cy = loc[:, 1] * v_c * priors[:, 3] + priors[:, 1]
        # sizes
        w  = priors[:, 2] * torch.exp(loc[:, 2] * v_s)
        h  = priors[:, 3] * torch.exp(loc[:, 3] * v_s)

        boxes = torch.stack([cx, cy, w, h], dim=1)

        # to xyxy
        # x1 = cx - 0.5 * w
        # y1 = cy - 0.5 * h
        # x2 = cx + 0.5 * w
        # y2 = cy + 0.5 * h
        # boxes = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes
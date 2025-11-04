import torch
from torch import nn
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

from collections import OrderedDict



class mySSD(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super(mySSD, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # image size should be 300x300
        self.img_h = 300
        self.img_w = 300

        # create priors
        self.priors = self.create_default_boxes() # size [8732, 4]

        # variances
        self.variance_center = 0.1
        self.variance_size = 0.2


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
# PyTorch built in SSD300 has a maxpool2d layer here, and padding=dilation=6 on the first conv2d layer
        self.extra_conv6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=6, dilation=6), # (B, 1024, 19, 19)
                nn.BatchNorm2d(num_features=1024),
                nn.ReLU(inplace=True)
            )

        self.extra_conv7 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),                       # (B, 1024, 19, 19)
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
                # nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True)
            )
        
        self.extra_conv11_2 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),                         # (B, 128, 3, 3)
                # nn.BatchNorm2d(num_features=128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),                         # (B, 256, 1, 1)
                # nn.BatchNorm2d(num_features=256),
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


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

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



    @staticmethod
    def create_default_boxes(s_min: float = 0.2, s_max: float = 0.9, clip: bool = True) -> torch.Tensor:
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

        # Example: aspect_ratios_per_level = 2
        # This will produce 4 default boxes (per center).
        # Given scales s, sp, create squares of with side length s and sp.
        # Create rectangles with scale s and aspect ratio 2, 1/2.
        # A total of 4 boxes are created (per center).

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
    


    # @torch.no_grad()
    # def predict(self,
    #             x: torch.Tensor,
    #             score_thresh: float = 0.05,
    #             nms_iou: float = 0.5,
    #             max_per_img: int = 200,
    #             class_agnostic: bool = False) -> List[torch.Tensor]:
    #     """
    #     Returns a list of length B (batch size).
    #     Each element is a (K_i, 6) tensor: [x1, y1, x2, y2, score, label],
    #     where label ∈ {1..C-1} corresponds to foreground classes (background dropped).
    #     """
    #     self.eval()
    #     loc_all, conf_all = self(x)              # (B,P,4), (B,P,C)
    #     device = loc_all.device
    #     B, P, C = conf_all.shape
    #     assert P == self.priors.size(0), "P mismatch with priors"
    #     assert C == self.num_classes and C >= 2, "num_classes must include background"

    #     # 1) class scores (drop background)
    #     scores_all = conf_all.softmax(dim=-1)[..., 1:]   # (B,P,C-1)
    #     num_fg = C - 1

    #     H, W = self.img_h, self.img_w
    #     outputs = []

    #     for b in range(B):
    #         # 2) decode using provided function: normalized cxcywh in [0,1]
    #         #    loc for this image: (P,4), priors: (P,4)
    #         boxes_cxcywh = self.decode_ssd(loc_all[b], self.priors.to(device), variances=(self.variance_center, self.variance_size))
    #         # 3) cxcywh -> pixel xyxy, clamp
    #         cx, cy, w, h = boxes_cxcywh.unbind(dim=1)
    #         x1 = (cx - 0.5 * w).clamp_(0, 1) * W
    #         y1 = (cy - 0.5 * h).clamp_(0, 1) * H
    #         x2 = (cx + 0.5 * w).clamp_(0, 1) * W
    #         y2 = (cy + 0.5 * h).clamp_(0, 1) * H
    #         b_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)       # (P,4)


    #         # 4) threshold scores, gather
    #         b_scores = scores_all[b]                                  # (P,num_fg)
    #         keep_mask = b_scores > score_thresh                       # (P,num_fg)
    #         if not keep_mask.any():
    #             outputs.append(torch.empty(0, 6, device=b_boxes_xyxy.device))
    #             continue

    #         pri_idx, cls_idx = keep_mask.nonzero(as_tuple=True)       # (M,) , cls_idx ∈ [0..C-2]
    #         sel_boxes  = b_boxes_xyxy[pri_idx]                        # (M,4)
    #         sel_scores = b_scores[pri_idx, cls_idx]                   # (M,)
    #         sel_labels = cls_idx                                      # (M,)

    #         # 5) NMS (per-class by default)
    #         if class_agnostic:
    #             keep = self.diou_nms(sel_boxes, sel_scores, diou_threshold=nms_iou)
    #         else:
    #             kept_list = []
    #             for cls in sel_labels.unique():
    #                 m = (sel_labels == cls)
    #                 if m.any():
    #                     idx_local = self.diou_nms(sel_boxes[m], sel_scores[m], diou_threshold=nms_iou)
    #                     # map back to original indices within sel_*
    #                     kept_list.append(m.nonzero(as_tuple=True)[0][idx_local])
    #             keep = torch.cat(kept_list, dim=0)
    #             # optional: re-sort by score if you want global top-K by confidence
    #             keep = keep[sel_scores[keep].argsort(descending=True)]

    #         keep = keep[:max_per_img]
    #         dets = torch.cat([
    #             sel_boxes[keep],
    #             sel_scores[keep].unsqueeze(1),
    #             (sel_labels[keep] + 1).to(sel_boxes.dtype).unsqueeze(1) # shift from 0, ..., C-2 to 1, ..., C-1
    #         ], dim=1)  # (K,6)

    #         outputs.append(dets)

    #         outputs.append({
    #         "labels": (sel_labels[keep] + 1).tolist(),
    #         "scores": sel_scores[keep].tolist(),
    #         "boxes": sel_boxes[keep]
    #         })

    #     return outputs
    


    @torch.no_grad()
    def predict(self,
                x: torch.Tensor,
                score_thresh: float = 0.05,
                nms_thresh: float = 0.5,
                max_per_img: int = 200,
                class_agnostic: bool = False,
            ) -> List[Dict[str, object]]:
        """
        Returns a list of length B; each element is a dict:
        {
            'labels': List[int],              # 1..C-1
            'scores': List[float],            # confidences
            'boxes' : Tensor[K,4] (xyxy)      # pixel coords on (H,W)
        }
        """
        self.eval()
        loc_all, conf_all = self(x)                      # (B,P,4), (B,P,C)
        B, P, C = conf_all.shape
        device = conf_all.device
        assert P == self.priors.size(0)
        assert C == self.num_classes and C >= 2

        # softmax and drop background
        scores_all = conf_all.softmax(dim=-1)[..., 1:]   # (B,P,C-1)
        num_fg = C - 1

        H, W = self.img_h, self.img_w
        v_c, v_s = self.variance_center, self.variance_size
        priors = self.priors.to(device)

        out: List[Dict[str, object]] = []

        for b in range(B):
            # decode to normalized cxcywh
            boxes_cxcywh = self.decode_ssd(loc_all[b], priors, variances=(v_c, v_s))  # (P,4)
            # cxcywh -> pixel xyxy + clamp
            cx, cy, w, h = boxes_cxcywh.unbind(dim=1)
            x1 = (cx - 0.5 * w).clamp(0, 1) * W
            y1 = (cy - 0.5 * h).clamp(0, 1) * H
            x2 = (cx + 0.5 * w).clamp(0, 1) * W
            y2 = (cy + 0.5 * h).clamp(0, 1) * H
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)                     # (P,4)

            b_scores = scores_all[b]                                              # (P,num_fg)
            keep_mask = b_scores > score_thresh                                   # (P,num_fg)

            if not keep_mask.any():
                out.append({
                    "labels": [],
                    "scores": [],
                    "boxes": boxes_xyxy.new_zeros((0, 4))
                })
                continue

            pri_idx, cls0_idx = keep_mask.nonzero(as_tuple=True)                  # 0-based class ids
            sel_boxes  = boxes_xyxy[pri_idx]                                       # (M,4)
            sel_scores = b_scores[pri_idx, cls0_idx]                               # (M,)
            sel_labels0 = cls0_idx                                                # (M,)

            # NMS
            if class_agnostic:
                keep = self.diou_nms(sel_boxes, sel_scores, diou_threshold=nms_thresh)
            else:
                kept = []
                for c in sel_labels0.unique():
                    m = (sel_labels0 == c)
                    if m.any():
                        idx_local = self.diou_nms(sel_boxes[m], sel_scores[m], diou_threshold=nms_thresh)
                        kept.append(m.nonzero(as_tuple=True)[0][idx_local])
                keep = torch.cat(kept, dim=0)
                keep = keep[sel_scores[keep].argsort(descending=True)]

            keep = keep[:max_per_img]

            # shift to 1..C-1 for output
            labels_out = (sel_labels0[keep] + 1).tolist()                          # List[int]
            scores_out = sel_scores[keep].tolist()                                 # List[float]
            boxes_out  = sel_boxes[keep]                                           # Tensor[K,4]

            out.append({
                "labels": labels_out,
                "scores": scores_out,
                "boxes": boxes_out
            })

        return out



    @staticmethod
    def diou_nms(boxes: torch.Tensor,
                 scores: torch.Tensor,
                 diou_threshold: float,
                 eps: float = 1e-9) -> torch.Tensor:
        """
        boxes:  (N,4) xyxy in absolute coords
        scores: (N,)
        returns indices of kept boxes (like torchvision.nms), sorted by score desc
        """
        if boxes.numel() == 0:
            return boxes.new_zeros((0,), dtype=torch.long)

        x1, y1, x2, y2 = boxes.unbind(dim=1)
        areas = (x2 - x1).clamp_(min=0) * (y2 - y1).clamp_(min=0)

        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break

            rest = order[1:]

            # Intersection
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            iw = (xx2 - xx1).clamp(min=0)
            ih = (yy2 - yy1).clamp(min=0)
            inter = iw * ih
            union = areas[i] + areas[rest] - inter + eps
            iou = inter / union

            # Center distance squared
            cx_i = (x1[i] + x2[i]) * 0.5
            cy_i = (y1[i] + y2[i]) * 0.5
            cx_j = (x1[rest] + x2[rest]) * 0.5
            cy_j = (y1[rest] + y2[rest]) * 0.5
            rho2 = (cx_i - cx_j)**2 + (cy_i - cy_j)**2

            # Enclosing box diagonal squared
            ex1 = torch.minimum(x1[i], x1[rest])
            ey1 = torch.minimum(y1[i], y1[rest])
            ex2 = torch.maximum(x2[i], x2[rest])
            ey2 = torch.maximum(y2[i], y2[rest])
            cw = (ex2 - ex1).clamp(min=0)
            ch = (ey2 - ey1).clamp(min=0)
            c2 = cw**2 + ch**2 + eps

            diou = iou - rho2 / c2

            # Keep boxes whose DIoU ≤ threshold (i.e., not “too similar + close”)
            mask = diou <= diou_threshold
            order = rest[mask]

        return torch.tensor(keep, device=boxes.device, dtype=torch.long)



    @staticmethod
    def encode_ssd(
        gt_boxes_cxcywh,          # [G,4] normalized (cx,cy,w,h)
        gt_labels,                # [G]
        priors_cxcywh,            # [P,4] normalized (cx,cy,w,h)
        iou_thresh=0.5,
        variances=(0.1, 0.2),
        background_class=0
    ):
        """
        Returns:
        loc_target: [P,4] (tx,ty,tw,th) per prior (positives encoded, negatives filled too)
        cls_target: [P]   background for negatives, matched GT label for positives
        pos_mask:   [P]   boolean positives
        matched_gt_xyxy: [P,4] GT boxes matched to each prior (xyxy, normalized)
        """
        device = priors_cxcywh.device
        dtype  = priors_cxcywh.dtype
        priors_cxcywh = priors_cxcywh.to(device=device, dtype=dtype)

        G = gt_boxes_cxcywh.shape[0]
        P = priors_cxcywh.shape[0]

        # Edge case: no GT in the image
        if G == 0:
            cls_target = torch.full((P,), background_class, dtype=gt_labels.dtype, device=device)
            loc_target = torch.zeros((P, 4), dtype=dtype, device=device)
            pos_mask   = torch.zeros((P,), dtype=torch.bool, device=device)
            matched_gt_cxcywh = torch.zeros((P, 4), dtype=dtype, device=device)
            return loc_target, cls_target, pos_mask, matched_gt_cxcywh

        # Convert GT to xyxy for IoU
        gt_boxes_cxcywh = gt_boxes_cxcywh.to(device=device, dtype=dtype)
        gt_boxes_xyxy   = box_convert(boxes=gt_boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy').clamp(0, 1)
        priors_xyxy     = box_convert(boxes=priors_cxcywh, in_fmt='cxcywh', out_fmt='xyxy').clamp(0, 1)

        # IoU and matching
        iou = box_iou(priors_xyxy, gt_boxes_xyxy)           # [P,G]
        # Force bipartite matches: each GT gets its best prior
        best_prior_per_gt = iou.argmax(dim=0)                # [G]
        iou[best_prior_per_gt, torch.arange(G, device=device)] = 2.0

        best_gt_per_prior  = iou.argmax(dim=1)               # [P]
        best_iou_per_prior = iou.gather(1, best_gt_per_prior.view(-1,1)).squeeze(1)
        pos_mask = best_iou_per_prior >= iou_thresh

        # matched_gt_xyxy  = gt_boxes_xyxy[best_gt_per_prior]  # [P,4]
        matched_gt_cxcywh = gt_boxes_cxcywh[best_gt_per_prior]  # [P,4]

        # Encode offsets (inverse of SSD decode)
        v_c, v_s = variances
        t_xy = (matched_gt_cxcywh[:, :2] - priors_cxcywh[:, :2]) / priors_cxcywh[:, 2:] / v_c
        t_wh = torch.log(
            (matched_gt_cxcywh[:, 2:] / priors_cxcywh[:, 2:]).clamp(min=1e-12)
        ) / v_s

        loc_target = torch.zeros_like(priors_cxcywh)
        loc_target[:, :2] = t_xy
        loc_target[:, 2:] = t_wh

        # Class targets
        gt_labels = gt_labels.to(device=device)
        matched_labels = gt_labels[best_gt_per_prior]        # [P]
        cls_target = torch.full((P,), background_class, dtype=matched_labels.dtype, device=device)
        cls_target[pos_mask] = matched_labels[pos_mask]

        return loc_target, cls_target, pos_mask, matched_gt_cxcywh


    @staticmethod
    def decode_ssd(loc, priors, variances=(0.1, 0.2)):
        """
        loc:   [num_priors, 4]  (tx, ty, tw, th)
        priors:[num_priors, 4]  (cx_a, cy_a, w_a, h_a), normalized [0,1]
        returns boxes_cxcywh normalized to [0,1], shape [num_priors, 4]
        """
        v_c, v_s = variances
        # centers
        cx = loc[:, 0] * v_c * priors[:, 2] + priors[:, 0]
        cy = loc[:, 1] * v_c * priors[:, 3] + priors[:, 1]
        # sizes
        w  = priors[:, 2] * torch.exp(loc[:, 2] * v_s)
        h  = priors[:, 3] * torch.exp(loc[:, 3] * v_s)

        boxes = torch.stack([cx, cy, w, h], dim=1)
        return boxes
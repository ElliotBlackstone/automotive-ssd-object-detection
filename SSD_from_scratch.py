import torch
from torch import nn
from torchvision.ops import box_convert, box_iou, distance_box_iou, complete_box_iou

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
    


    


    @torch.no_grad()
    def predict(self,
                images: torch.Tensor,
                score_thresh: float = 0.05,
                nms_thresh: float = 0.5,
                max_per_img: int = 200,
                class_agnostic: bool = False,
                pre_loc_all: torch.Tensor | None = None,
                pre_conf_all: torch.Tensor | None = None,
                ) -> List[Dict[str, object]]:
        """
        Inputs
        images: Tensor of size [B, 3, 300, 300]
        score_thresh: Float between 0 and 1 determining the score threshold for kept predictions
        nms_thresh: Float between 0 and 1 determining the non-maximum suppression threshold
        max_per_img: Integer denoting the max amount of predictions per image
        class_agnostic: Boolean
        pre_loc_all: Tensor of size [B, P, 4], pre computation of loc_all, _ = self(images)
        pre_conf_all: Tensor of size [B, P, C], pre computation of _, conf_all = self(images)

        Output
        List of length B; each element is a dict:
        {
            'labels': Tensor, contains values 0, ..., C-2
            'scores': Tensor, contains confidences for each class
            'boxes' : Tensor of size [K,4] in 'xyxy' format
        }
        (B - batch size, P - number of priors (8732), C - number of classes)
        """

        # make sure score, nms threshold are valid
        if not (0.0 <= score_thresh < 1.0):
            raise ValueError(f"Score threshold should be greater than 0 and less than 1, recieved {score_thresh}.")
        
        if not (0.0 < nms_thresh < 1.0):
            raise ValueError(f"NMS threshold should be greater than 0 and less than 1, recieved {nms_thresh}.")
        
        self.eval()
        if (pre_loc_all is not None) & (pre_conf_all is not None):
            loc_all = pre_loc_all
            conf_all = pre_conf_all
        else:
            loc_all, conf_all = self(images)                      # (B,P,4), (B,P,C)
        
        B, P, C = conf_all.shape
        device = conf_all.device
        assert P == self.priors.size(0)
        assert C == self.num_classes and C >= 2

        # softmax and drop background
        scores_all = conf_all.softmax(dim=-1)[..., 1:]   # (B,P,C-1)

        H, W = self.img_h, self.img_w
        v_c, v_s = self.variance_center, self.variance_size
        priors = self.priors
        if priors.device != device:
            priors = priors.to(device)

        out: List[Dict[str, object]] = []

        for b in range(B):
            # scores for this image: [P,num_fg]
            b_scores = scores_all[b]

            # threshold BEFORE decoding
            keep_mask = b_scores > score_thresh  # [P,num_fg]
            if not keep_mask.any():
                out.append({
                    "labels": torch.empty(0, dtype=torch.int64, device=device),
                    "scores": torch.empty(0, dtype=torch.float32, device=device),
                    "boxes":  priors.new_zeros((0, 4))
                })
                continue
            # indices of priors and class-ids that survive threshold
            # pri_idx: [M], cls0_idx: [M]
            pri_idx, cls0_idx = keep_mask.nonzero(as_tuple=True)

            # slice loc + priors to those M priors
            loc_sel    = loc_all[b, pri_idx]  # [M,4] offsets for kept priors
            priors_sel = priors[pri_idx]      # [M,4] priors for kept priors

            # decode only these M priors to normalized cxcywh
            boxes_cxcywh = self.decode_ssd(loc=loc_sel,
                                           priors=priors_sel,
                                           variances=(v_c, v_s))  # [M,4]

            # decode to normalized cxcywh
            # boxes_cxcywh = self.decode_ssd(loc_all[b], priors, variances=(v_c, v_s))  # (P,4)
            # cxcywh -> pixel xyxy + clamp
            cx, cy, w, h = boxes_cxcywh.unbind(dim=1)
            x1 = (cx - 0.5 * w).clamp(0, 1) * W
            y1 = (cy - 0.5 * h).clamp(0, 1) * H
            x2 = (cx + 0.5 * w).clamp(0, 1) * W
            y2 = (cy + 0.5 * h).clamp(0, 1) * H
            # boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)                     # (P,4)

            sel_boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [M,4]

            # scores / labels for these kept (prior, class) pairs
            sel_scores  = b_scores[pri_idx, cls0_idx]  # [M]
            sel_labels0 = cls0_idx                     # [M], 0-based foreground labels

            # b_scores = scores_all[b]                                              # (P,num_fg)
            # keep_mask = b_scores > score_thresh                                   # (P,num_fg)

            # if not keep_mask.any():
            #     out.append({
            #         "labels": torch.empty(0, dtype=torch.int64, device=device),
            #         "scores": torch.empty(0, dtype=torch.float32, device=device),
            #         "boxes": boxes_xyxy.new_zeros((0, 4))
            #     })
            #     continue

            # pri_idx, cls0_idx = keep_mask.nonzero(as_tuple=True)                   # 0-based class ids
            # sel_boxes  = boxes_xyxy[pri_idx]                                       # (M,4)
            # sel_scores = b_scores[pri_idx, cls0_idx]                               # (M,)
            # sel_labels0 = cls0_idx                                                 # (M,)

            # NMS
            if class_agnostic:
                keep = self.iou_nms(sel_boxes, sel_scores, iou_threshold=nms_thresh)
                # ensure highest-score first before truncation
                keep = keep[sel_scores[keep].argsort(descending=True)]
            else:
                # kept = []
                # for c in sel_labels0.unique():
                #     m = (sel_labels0 == c)
                #     if m.any():
                #         idx_local = self.iou_nms(sel_boxes[m], sel_scores[m], iou_threshold=nms_thresh)
                #         kept.append(m.nonzero(as_tuple=True)[0][idx_local])
                # keep = torch.cat(kept, dim=0)
                kept = []
                for c in sel_labels0.unique():
                    idx_c = (sel_labels0 == c).nonzero(as_tuple=True)[0]
                    if idx_c.numel() == 0:
                        continue
                    idx_local = self.iou_nms(sel_boxes[idx_c], sel_scores[idx_c], iou_threshold=nms_thresh)
                    kept.append(idx_c[idx_local])
                keep = torch.cat(kept, dim=0)
                keep = keep[sel_scores[keep].argsort(descending=True)]

            keep = keep[:max_per_img]

            
            labels_out = sel_labels0[keep]                                       # Tensor
            scores_out = sel_scores[keep]                                        # Tensor
            boxes_out  = sel_boxes[keep]                                         # Tensor[K,4]

            out.append({
                "labels": labels_out,
                "scores": scores_out,
                "boxes": boxes_out
            })

        return out

    
    @staticmethod
    def iou_nms(boxes: torch.Tensor,
                scores: torch.Tensor,
                iou_threshold: float,
                ) -> torch.Tensor:
        """
        boxes:  (N,4) xyxy, strictly x1<x2, y1<y2 (as torchvision expects)
        scores: (N,)
        returns LongTensor indices of kept boxes (sorted by score desc)
        """
        if boxes.numel() == 0:
            return boxes.new_zeros((0,), dtype=torch.long)

        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]

            # pairwise CIoU between the top-1 box and the remaining boxes
            iou = complete_box_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
            # suppress boxes "too similar & close" to the current top box
            order = rest[iou <= iou_threshold]

        return torch.tensor(keep, device=boxes.device, dtype=torch.long)



    @staticmethod
    def encode_ssd(gt_boxes_cxcywh: torch.Tensor,          # [G,4] normalized (cx,cy,w,h)
                   gt_labels: torch.Tensor,                # [G]
                   priors_cxcywh: torch.Tensor,            # [P,4] normalized (cx,cy,w,h)
                   iou_thresh: float = 0.5,
                   variances: Tuple[float, float] = (0.1, 0.2),
                   background_class: int = 0
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs
        gt_boxes_cxcywh: Ground truth (GT) bounding boxes tensor in 'cxcywh' format
        gt_labels: Tensor containing labels (0, 1, ..., C-2, where C is the total
                   number of classes, including background) corresponding to GT boxes
        priors_cxcywh: Tensor of shape [P, 4] containing all priors in 'cxcywh' format
        variances:
        background_class: integer denoting background class, must be 0

        Returns:
        loc_target: [P,4] (tx,ty,tw,th) per prior (positives encoded, negatives filled too)
        cls_target: [P]   background for negatives, matched GT label for positives
        pos_mask:   [P]   boolean positives
        matched_gt_xyxy: [P,4] GT boxes matched to each prior (xyxy, normalized)
        (P is the number of priors, 8732)
        """

        # only works if background_class = 0
        if background_class != 0:
            raise ValueError(f"Background should be 0, recieved {background_class}.")
        
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
        iou = complete_box_iou(priors_xyxy, gt_boxes_xyxy)           # [P,G]
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
        cls_target[pos_mask] = matched_labels[pos_mask] + 1  # shift by 1 because 0 is reserved for 'background'

        return loc_target, cls_target, pos_mask, matched_gt_cxcywh


    @staticmethod
    def decode_ssd(loc: torch.Tensor,
                   priors: torch.Tensor,
                   variances: Tuple[float, float] = (0.1, 0.2),
                   ) -> torch.Tensor:
        """
        Inputs
        loc: Tensor of shape [P, 4] containing (tx, ty, tw, th)
        priors: Priors of shape [P, 4] containing (cx_a, cy_a, w_a, h_a), normalized [0,1]
        variances: Tuple containing two positive floats, default (0.1, 0.2)

        Outputs
        boxes_cxcywh normalized to [0,1], shape [P, 4]
        (P is the number of priors, 8732)
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
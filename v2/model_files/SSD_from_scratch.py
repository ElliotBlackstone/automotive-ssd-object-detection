import torch
from torch import nn
from torchvision.ops import box_convert
import gen_nms

from typing import Tuple, Dict, List

from collections import OrderedDict

from .create_default_boxes import create_default_boxes
from .decode_ssd import decode_ssd
from .nms_variant import run_nms_by_variant



class mySSD(nn.Module):
    def __init__(self,
                 class_to_idx_dict: Dict,
                 in_channels: int = 3,
                 variances: Tuple[float, float] = (0.1, 0.2)):
        
        super(mySSD, self).__init__()

        self.in_channels = in_channels
        self.class_to_idx = class_to_idx_dict
        self.idx_to_class = {v: k for k, v in class_to_idx_dict.items()}
        self.num_classes = len(class_to_idx_dict) + 1 # add 1 for background

        # image size should be 300x300
        self.img_h = 300
        self.img_w = 300

        # create priors
        priors = create_default_boxes() # size [8732, 4]
        self.register_buffer("priors", priors, persistent=False)
        priors_xyxy = box_convert(priors, in_fmt='cxcywh', out_fmt='xyxy').clamp(0, 1)
        self.register_buffer("priors_xyxy", priors_xyxy, persistent=False)

        # variances
        self.variance_center, self.variance_size = variances


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
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),   # applied to VGG16_UpTo_conv4_3 - output size: (B, 4*4, 38, 38)
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),  # applied to extra_conv7        - output size: (B, 6*4, 19, 19)
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),   # applied to extra_conv8_2      - output size: (B, 6*4, 10, 10)
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),   # applied to extra_conv9_2      - output size: (B, 6*4, 5, 5)
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),   # applied to extra_conv10_2     - output size: (B, 4*4, 3, 3)
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)    # applied to extra_conv11_2     - output size: (B, 4*4, 1, 1)
        ])

        self.cls_head = nn.ModuleList([
            nn.Conv2d(512, 4 * self.num_classes, kernel_size=3, padding=1),  # applied to VGG16_UpTo_conv4_3 - output size: (B, 4*num_classes, 38, 38)
            nn.Conv2d(1024, 6 * self.num_classes, kernel_size=3, padding=1), # applied to extra_conv7        - output size: (B, 6*num_classes, 19, 19)
            nn.Conv2d(512, 6 * self.num_classes, kernel_size=3, padding=1),  # applied to extra_conv8_2      - output size: (B, 6*num_classes, 10, 10)
            nn.Conv2d(256, 6 * self.num_classes, kernel_size=3, padding=1),  # applied to extra_conv9_2      - output size: (B, 6*num_classes, 5, 5)
            nn.Conv2d(256, 4 * self.num_classes, kernel_size=3, padding=1),  # applied to extra_conv10_2     - output size: (B, 4*num_classes, 3, 3)
            nn.Conv2d(256, 4 * self.num_classes, kernel_size=3, padding=1)   # applied to extra_conv11_2     - output size: (B, 4*num_classes, 1, 1)
        ])

        # total detections per class: 4*38*38 + 6*19*19 + 6*10*10 + 6*5*5 + 4*3*3 + 4*1*1 = 8732


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)

        x = self.VGG16_UpTo_conv4_3(x)
        x_conv43 = x

        x = self.VGG16_extras(x)
        x = self.extra_conv6(x)

        x_conv7  = self.extra_conv7(x)
        x_conv8  = self.extra_conv8_2(x_conv7)
        x_conv9  = self.extra_conv9_2(x_conv8)
        x_conv10 = self.extra_conv10_2(x_conv9)
        x_conv11 = self.extra_conv11_2(x_conv10)

        feats = [x_conv43, x_conv7, x_conv8, x_conv9, x_conv10, x_conv11]

        loc_out = []
        cls_out = []

        for feat, box_head, cls_head in zip(feats, self.box_head, self.cls_head):
            loc = box_head(feat).permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            cls = cls_head(feat).permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            loc_out.append(loc)
            cls_out.append(cls)

        loc_bbox_form = torch.cat(loc_out, dim=1)   # [B, 8732, 4]
        cls_preds     = torch.cat(cls_out, dim=1)   # [B, 8732, num_classes]

        return loc_bbox_form, cls_preds

    


    @torch.inference_mode()
    def predict(self,
                images: torch.Tensor,
                score_thresh: float = 0.2,
                nms_thresh: float = 0.5,
                iou_variant: str = "IoU",
                max_per_img: int = 100,
                class_agnostic: bool = False,
                pre_loc_all: torch.Tensor | None = None,
                pre_conf_all: torch.Tensor | None = None,
                ) -> List[Dict[str, object]]:
        """
        Inputs
        images: Tensor of size [B, 3, 300, 300]
        score_thresh: Float between 0 and 1 determining the score threshold for kept predictions
        nms_thresh: Float between 0 and 1 determining the non-maximum suppression threshold
        iou_variant: string that must be on of "IoU", "GIoU", "DIoU", "CIoU"
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
        
        if (pre_loc_all is None) != (pre_conf_all is None):
            raise ValueError("Provide both pre_loc_all and pre_conf_all, or neither.")

        # due to check above pre_loc_all not being none implies we have pre_conf_all as well
        if pre_loc_all is not None:
            loc_all, conf_all = pre_loc_all, pre_conf_all
        else:
            loc_all, conf_all = self(images)                      # (B,P,4), (B,P,C)
        
        B, P, C = conf_all.shape
        device = conf_all.device

        if P != self.priors.size(0):
            raise ValueError(f"Expected {self.priors.size(0)} priors, got {P}.")
        if C != self.num_classes or C < 2:
            raise ValueError(f"Expected num_classes={self.num_classes} >= 2, got {C}.")

        H, W = self.img_h, self.img_w
        v_c, v_s = self.variance_center, self.variance_size
        priors = self.priors


        out: List[Dict[str, object]] = []

        for b in range(B):
            logits = conf_all[b]
            fg_logits = logits[:, 1:]                         # [P, C-1]
            best_fg_logit, best_label0 = fg_logits.max(dim=1) # one class per prior

            log_denom = torch.logsumexp(logits, dim=1)        # includes background
            best_scores = (best_fg_logit - log_denom).exp()   # true softmax prob for best fg class


            # threshold BEFORE decoding
            keep = best_scores > score_thresh  # [P]
            if not keep.any():
                out.append({
                    "labels": torch.empty(0, dtype=torch.int64, device=device),
                    "scores": torch.empty(0, dtype=torch.float32, device=device),
                    "boxes":  priors.new_zeros((0, 4))
                })
                continue

            # slice loc + priors to those M priors
            loc_sel    = loc_all[b, keep]  # [M,4] offsets for kept priors
            priors_sel = priors[keep]      # [M,4] priors for kept priors

            # decode only these M priors to normalized cxcywh
            boxes_cxcywh = decode_ssd(loc=loc_sel, priors=priors_sel, variances=(v_c, v_s))  # [M,4]

            cx, cy, w, h = boxes_cxcywh.unbind(dim=1)
            x1 = (cx - 0.5 * w).clamp(0, 1) * W
            y1 = (cy - 0.5 * h).clamp(0, 1) * H
            x2 = (cx + 0.5 * w).clamp(0, 1) * W
            y2 = (cy + 0.5 * h).clamp(0, 1) * H
            sel_boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [M,4]

            # scores / labels
            sel_scores  = best_scores[keep]  # [M]
            sel_labels0 = best_label0[keep]  # [M], 0-based foreground labels

            # NMS
            keep_nms = run_nms_by_variant(boxes=sel_boxes,
                                          scores=sel_scores,
                                          nms_thresh=nms_thresh,
                                          variant=iou_variant,
                                          class_agnostic=class_agnostic,
                                          idxs=None if class_agnostic else sel_labels0,
                                          )

            keep_nms = keep_nms[:max_per_img]


            out.append({"labels": sel_labels0[keep_nms],
                        "scores": sel_scores[keep_nms],
                        "boxes": sel_boxes[keep_nms]})

        return out

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image
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



class mySSD(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super(mySSD, self).__init__()

        self.in_channels = in_channels


# image size must be 300x300
#################### begin VGG16 model ####################        
        
        self.VGG16_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, in_channels, 300, 300) -> (B, 64, 300, 300)
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(inplace=True),                       
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 64, 300, 300) -> (B, 64, 300, 300)
            nn.BatchNorm2d(num_features=64),                   # no size change
            nn.ReLU(inplace=True)                              # no size change
        )

        self.VGG16_mp1 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 64, 300, 300) -> (B, 64, 150, 150)

        self.VGG16_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 64, 150, 150) -> (B, 128, 150, 150)
            nn.BatchNorm2d(num_features=128), 
            nn.ReLU(inplace=True),                        
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 128, 150, 150) -> (B, 128, 150, 150)
            nn.BatchNorm2d(num_features=128), 
            nn.ReLU(inplace=True)                         
        )

        self.VGG16_mp2 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 128, 150, 150) -> (B, 128, 75, 75)

        self.VGG16_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 128, 75, 75) -> (B, 256, 75, 75)
            nn.BatchNorm2d(num_features=256), 
            nn.ReLU(inplace=True),                        
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 256, 75, 75) -> (B, 256, 75, 75)
            nn.BatchNorm2d(num_features=256), 
            nn.ReLU(),                        
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 256, 75, 75) -> (B, 256, 75, 75)
            nn.BatchNorm2d(num_features=256), 
            nn.ReLU(inplace=True)                         
        )

        self.VGG16_mp3 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 256, 75, 75) -> (B, 256, 38, 38)

        self.VGG16_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 256, 38, 38) -> (B, 512, 38, 38)
            nn.BatchNorm2d(num_features=512), 
            nn.ReLU(inplace=True),                        
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 512, 38, 38) -> (B, 512, 38, 38)
            nn.BatchNorm2d(num_features=512), 
            nn.ReLU(inplace=True),                        
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 512, 38, 38) -> (B, 512, 38, 38)
            nn.BatchNorm2d(num_features=512), 
            nn.ReLU(inplace=True)                         
        )

        self.VGG16_mp4 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 512, 38, 38) -> (B, 512, 19, 19)

        self.VGG16_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 512, 19, 19) -> (B, 512, 19, 19)
            nn.BatchNorm2d(num_features=512), 
            nn.ReLU(inplace=True),                        
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 512, 19, 19) -> (B, 512, 19, 19)
            nn.BatchNorm2d(num_features=512), 
            nn.ReLU(inplace=True),                        
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),                              # (B, 512, 19, 19) -> (B, 512, 19, 19)
            nn.BatchNorm2d(num_features=512), 
            nn.ReLU(inplace=True)                         
        )

#################### end VGG16 model ####################



# Additional layers for SSD
        # the first fc6/conv6 layer seems to be a typo - start with Conv7/FC7
        # PyTorch built in SSD300 has a maxpool2d layer here, and padding=stride=6 on the first conv2d layer
        self.extra_conv7 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, dilation=1), # (B, 1024, 19, 19)
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
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),   # applied to VGG16_layer4   - output size: (B, 16, 38, 38)
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),  # applied to extra_conv7    - output size: (B, 24, 19, 19)
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),   # applied to extra_conv8_2  - output size: (B, 24, 10, 10)
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),   # applied to extra_conv9_2  - output size: (B, 24, 5, 5)
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),   # applied to extra_conv10_2 - output size: (B, 16, 3, 3)
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)    # applied to extra_conv11_2 - output size: (B, 16, 1, 1)
        ])

        self.cls_head = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),  # applied to VGG16_layer4   - output size: (B, 4*classes, 38, 38)
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1), # applied to extra_conv7    - output size: (B, 6*classes, 19, 19)
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),  # applied to extra_conv8_2  - output size: (B, 6*classes, 10, 10)
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),  # applied to extra_conv9_2  - output size: (B, 6*classes, 5, 5)
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),  # applied to extra_conv10_2 - output size: (B, 4*classes, 3, 3)
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)   # applied to extra_conv11_2 - output size: (B, 4*classes, 1, 1)
        ])

        # total detections per class: 4*38*38 + 6*19*19 + 6*10*10 + 6*5*5 + 4*3*3 + 4*1*1 = 8732
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class ResNet(nn.Module):
    def __init__(self, n_classes=4, pretrained=True, hidden_size=2048, model_file=None, version=50, feature_extract=True):
        super().__init__()
        if version == 18: 
            self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif version == 50:
            self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.require_all_grads(feature_extract)
        # self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        # if not pretrained:  
        #     # self.resnet.fc = nn.Linear(hidden_size, 365)
        #     # checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        #     # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        #     # self.resnet.load_state_dict(state_dict)
        self.resnet.fc = nn.Linear(hidden_size, n_classes)

    def require_all_grads(self, feature_extract=True):
        for name, param in self.resnet.named_parameters():
            if 'fc' in name: 
                param.requires_grad = True
            else: 
                param.requires_grad = feature_extract
            if param.requires_grad:
                print(name, param.requires_grad)

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs

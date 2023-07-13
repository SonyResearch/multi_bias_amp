import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from model_arc import resnet, lenet

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class multilabel_classifier():
    def __init__(self, device, dtype, model_name, nclasses=1, modelpath=None, hidden_size=2048, pretrained=True, version=50, feature_extract=True):
        self.nclasses = nclasses
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        if model_name == 'resnet':
            self.model = resnet.ResNet(n_classes=nclasses, hidden_size=hidden_size, pretrained=pretrained, version=version, feature_extract=feature_extract)
        elif model_name == 'lenet': 
            self.model = lenet.LeNet(nclasses)

        # Multi-GPU training
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.epoch = 1
        self.print_freq = 100

        if modelpath != None:
            A = torch.load(modelpath, map_location=device)
            load_state_dict = A['model']
            load_prefix = list(load_state_dict.keys())[0][:6]
            new_state_dict = {}
            for key in load_state_dict:
                value = load_state_dict[key]
                if load_prefix == 'module':
                    new_key = key[7:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            self.model.load_state_dict(new_state_dict)
            self.epoch = A['epoch']

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def save_model(self, path):
        torch.save({'model':self.model.state_dict(), 'optim':self.optimizer, 'epoch':self.epoch}, path)

    def train(self, loader):
        """Train the 'standard baseline' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):
            images = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)
            self.optimizer.zero_grad()
            outputs = self.forward(images)
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)
        return loss_list

    def test(self, loader):
        """Evaluate the 'standard baseline' model"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        with torch.no_grad():
            files_list = []
            labels_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            scores_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            loss_list = []

            for i, (images, labels, ids) in enumerate(loader):
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)
                outputs = self.forward(images)
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(outputs.squeeze(), labels)
                loss_list.append(loss.item())
                scores = torch.sigmoid(outputs).squeeze()
                labels_list = np.concatenate((labels_list.squeeze(), labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list.squeeze(), scores.detach().cpu().numpy()), axis=0)
                files_list.extend(ids)
                if self.print_freq and (i % self.print_freq == 0):
                    print('Validation epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)
            self.epoch += 1
        return labels_list, scores_list, loss_list, files_list

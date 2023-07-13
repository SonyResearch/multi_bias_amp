import pickle
import glob
import time
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage import transform

class CustomDataset(Dataset):
    def __init__(self, img_labels, transform=T.ToTensor()):
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        labels = self.img_labels[index]
        ID = labels[0]
        img = Image.open(ID).convert('RGB')
        y = np.array([int(i) for i in labels[1:]])
        X = self.transform(img)
        return X, y, ID

def create_dataset(name, labels_path, B=32, train=True):

    img_labels = pickle.load(open(labels_path, 'rb'))

    if name == 'mnist':
        normalize = T.Normalize((0.1307,), (0.3081,))
        if train: 
            transform = T.Compose([
                T.ToTensor(),
                normalize
            ])
        else: 
            transform = T.Compose([
                T.ToTensor(),
                normalize
            ])
    elif name == 'cifar':
        normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        if train: 
            transform = T.Compose([
                T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.Pad(16),
                T.ToTensor(),
                normalize
            ])
        else: 
            transform = T.Compose([
                T.Pad(16),
                T.ToTensor(),
                normalize
            ])
    else:
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            transform = T.Compose([
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])

    dset = CustomDataset(img_labels, transform)
    loader = DataLoader(dset, batch_size=B, shuffle=train, num_workers=1)
    return loader
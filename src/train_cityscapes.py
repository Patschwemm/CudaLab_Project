import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torchvision

import os
import sys
sys.path.insert(0, "../utils")
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

import cityscapes_loader
import Temporal_UNET_Template
import train_eval
import importlib
import temporal_modules

if __name__ == "__main__":
    is_sequence = True

    dataset_root_dir = "/home/nfs/inf6/data/datasets/cityscapes/"

    train_ds = cityscapes_loader.cityscapesLoader(root=dataset_root_dir, split='train', img_size=(512, 1024), is_transform=True, is_sequence=is_sequence)
    val_ds = cityscapes_loader.cityscapesLoader(root=dataset_root_dir, split='val', img_size=(512, 1024), is_transform=True, is_sequence=is_sequence)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False, drop_last=True)

    config = Temporal_UNET_Template.Temporal_UNetConfig(temporal_cell=temporal_modules.Conv2dRNNCell ,out_channels=train_ds.n_classes+1)

    temp_unet = Temporal_UNET_Template.Temporal_UNet(config)

    temp_unet_optim = torch.optim.Adam(temp_unet.parameters(), lr=3e-4)

    criterion = nn.CrossEntropyLoss()

    
    epochs=10
    temp_unet_trainer = train_eval.Trainer(
            temp_unet, temp_unet_optim, criterion, 
            train_loader, valid_loader, "coco", epochs, 
            sequence=False, all_labels=91, start_epoch=0)
    
    temp_unet_trainer.train_model()
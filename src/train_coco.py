import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
# need ms coco api to get dataset
from pycocotools import mask as mask
import numpy as np
import matplotlib.pyplot as plt
import utils
import utils.train_eval


import coco_dataset
import importlib
import Temporal_UNET_Template

if __name__ == "__main__":
    utils.set_random_seed()

    
    # path to MS COCO dataset
    train_data_dir = '/home/nfs/inf6/data/datasets/coco/train2017'
    val_data_dir = '/home/nfs/inf6/data/datasets/coco/val2017'


    # initialize COCO API for segmentation
    train_ann_file = '/home/nfs/inf6/data/datasets/coco/annotations/instances_train2017.json'
    val_ann_file = '/home/nfs/inf6/data/datasets/coco/annotations/instances_val2017.json'


    # make dataset, default transforms is resizing and to tensor
    train_set = coco_dataset.Coco_Dataset(train_data_dir, train_ann_file, mode="segmentation")
    val_set = coco_dataset.Coco_Dataset(val_data_dir, val_ann_file, mode="segmentation")

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    config = Temporal_UNET_Template.Temporal_UNetConfig()

    temp_unet = Temporal_UNET_Template.Temporal_UNet(config)

    temp_unet_optim = torch.optim.Adam(temp_unet.parameters(), lr=3e-4)

    criterion = nn.CrossEntropyLoss()

    
    epochs=20
    temp_unet_trainer = utils.train_eval.Trainer(
        temp_unet, temp_unet_optim, criterion, train_loader, valid_loader, "coco", epochs, sequence=False, all_labels=91, start_epoch=10)
    
    temp_unet_trainer.train_model()
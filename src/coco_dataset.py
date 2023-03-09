import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

# code snippets from https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5

class Coco_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None, target_transforms=None):
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # get masks and according labels 
        masks, labels = [], []
        for i in range(num_objs):
            masks.append(coco.annToMask(ann=coco_annotation[i]))
            labels.append(coco_annotation[i]['category_id'])
            
        masks = np.array(masks)
        masks = torch.tensor(masks)
        labels = torch.tensor(labels)


        if self.transforms is not None:
            img = self.transforms(img)
            for i in range(len(masks)):
                masks[i] = self.transforms(masks)
            
        # Annotation is in dictionary format
        label_dict = {}
        label_dict["masks"] = masks
        label_dict["labels"] = labels

        return img, label_dict

    def __len__(self):
        return len(self.ids)
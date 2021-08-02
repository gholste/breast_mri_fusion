import os
from copy import deepcopy

import cv2
import numpy as np

import torch
from torchvision import transforms
from albumentations.augmentations.transforms import Blur, HorizontalFlip, ElasticTransform, RandomScale, Resize, Rotate, RandomContrast
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch.transforms import ToTensor

class BreastMRIFusionDataset(torch.utils.data.Dataset):
    def __init__(self, fpath, augment=False, n_TTA=0, n=None):
        self.fpath = fpath
        self.augment = augment
        self.n_TTA = n_TTA
        self.n = n

        if self.augment:
            self.transform = Compose([
                HorizontalFlip(p=0.5),
                OneOf([Blur(blur_limit=4, p=0.5), RandomContrast(p=0.5)], p=1),
                ElasticTransform(alpha_affine=10, sigma=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                Compose([RandomScale(scale_limit=0.2, p=1), Resize(224, 224, p=1)], p=0.5),
                Rotate(20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                ToTensor()
             ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        if self.n_TTA != 0:
            self.tta_transform = Compose([
                HorizontalFlip(p=0.5),
                OneOf([Blur(blur_limit=4, p=0.5), RandomContrast(p=0.5)], p=1),
                Rotate(20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                ToTensor()
            ])

        self.meta_features = np.load(os.path.join(self.fpath, "1_meta.npy")).shape[0]

    def __len__(self):
        if self.n is not None:
            return self.n
        else:
            return len([x for x in os.listdir(self.fpath) if "_x" in x])

    def __getitem__(self, idx):
        x    = np.load(os.path.join(self.fpath, str(idx + 1) + "_x.npy"))
        meta = np.load(os.path.join(self.fpath, str(idx + 1) + "_meta.npy"))
        y    = np.load(os.path.join(self.fpath, str(idx + 1) + "_y.npy"))
     
        if self.n_TTA != 0:
            x = np.stack([self.tta_transform(image=x)["image"].float() for _ in range(self.n_TTA)], axis=-1)
        else:
            if self.augment:
                x = self.transform(image=x)["image"].float()
            else:
                x = self.transform(x).float()
        meta = torch.Tensor(meta).float()
        y = torch.Tensor(y).float()

        return {"image": x, "metadata": meta, "label": y}

class FeatureImpDataset(torch.utils.data.Dataset):
    def __init__(self, fpath):
        self.fpath = fpath

        self.n = len([x for x in os.listdir(self.fpath) if "_x" in x])
        self.transform = transforms.ToTensor()

        self.x_test = np.array([np.load(os.path.join(self.fpath, str(i+1) + "_x.npy")) for i in range(self.n)])
        self.orig_meta_test = np.array([np.load(os.path.join(self.fpath, str(i+1) + "_meta.npy")) for i in range(self.n)])
        self.meta_test = deepcopy(self.orig_meta_test)
        self.y_test = np.array([np.load(os.path.join(self.fpath, str(i+1) + "_y.npy")) for i in range(self.n)])
	
        self.meta_features = self.orig_meta_test.shape[1]

    def __len__(self):
        return self.n

    def __getitem__(self, idx, orig=False):
        x    = self.x_test[idx]
        meta = self.meta_test[idx]
        y    = self.y_test[idx]

        x    = self.transform(x).float()
        meta = torch.Tensor(meta).float()
        y    = torch.Tensor(y).float()

        return {"image": x, "metadata": meta, "label": y}
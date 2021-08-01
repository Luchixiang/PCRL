import copy
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PCRLChestPretask(Dataset):
    def __init__(self, config, img_train, train, transform=None, spatial_transform=None):
        self.config = config
        self.imgs = img_train
        self.train = train
        self.transform = transform
        self.spatial_transform = spatial_transform
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize_trans = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_name = self.imgs[index]
        y = Image.open(image_name).convert('RGB')
        y1 = self.spatial_transform(y)
        y2 = self.spatial_transform(y)
        gt1 = copy.deepcopy(y1)
        gt2 = copy.deepcopy(y2)
        gt1 = self.normalize_trans(gt1)
        gt2 = self.normalize_trans(gt2)
        input1 = self.transform(y1)
        input2 = self.transform(y2)
        mask1 = copy.deepcopy(input1)
        mask2 = copy.deepcopy(input2)
        mask1, aug_tensor1 = self.aug(mask1)
        mask2, aug_tensor2 = self.aug(mask2)

        return input1, input2, mask1, mask2, gt1, gt2, aug_tensor1, aug_tensor2

    def aug(self, img):
        img = img.numpy()
        aug_tensor = [0 for _ in range(6)]
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            aug_tensor[0] = 1
        if random.random() < 0.5:
            img = np.flip(img, axis=2)
            aug_tensor[1] = 1
        times = int(random.random() // 0.25)
        img = np.rot90(img, k=times, axes=(1, 2))
        aug_tensor[times + 2] = 1
        return torch.tensor(img.copy(), dtype=torch.float), torch.tensor(aug_tensor)

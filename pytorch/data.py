from torch.utils.data import DataLoader

from datasets import *
from utils import *
from torchvision import transforms, datasets
import torch
import torchio.transforms


class DataGenerator:

    def __init__(self, config):
        self.config = config

    def pcrl_luna_pretask(self):
        print('using the reverse_aug pretrain on luna')
        config = self.config
        dataloader = {}
        train_fold = [0, 1, 2, 3, 4, 5, 6, ]
        valid_fold = [7, 8, 9]
        file_list = get_luna_pretrain_list(config.ratio, config.root, train_fold)
        x_train, x_valid, _ = get_luna_list(config, train_fold, valid_fold, valid_fold, suffix='_img_',
                                            file_list=file_list)
        print(f'total train images {len(x_train)}, valid images {len(x_valid)}')
        transforms = [torchio.transforms.RandomFlip(),
                      torchio.transforms.RandomAffine(),
                      torchio.transforms.RandomBlur(),
                      ]
        transforms = torchio.transforms.Compose(transforms)
        train_ds = PCRLLunaPretask(config, x_train, train=True, transform=transforms)
        valid_ds = PCRLLunaPretask(config, x_valid, train=False)
        dataloader['train'] = DataLoader(train_ds, batch_size=config.batch_size,
                                         pin_memory=True, shuffle=True, num_workers=config.workers)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=config.batch_size,
                                        pin_memory=True, shuffle=False, num_workers=config.workers)
        dataloader['test'] = dataloader['eval']
        return dataloader

    def pcrl_chest_pretask(self):
        print('using reverse aug on chest')
        conf = self.config

        image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        spatial_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1)),
        ])
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform.transforms.append(Cutout(n_holes=3, length=32))
        train_file = './train_val_txt/chest_train.txt'
        train_imgs, train_labels = get_chest_list(train_file, conf.data)
        train_imgs = train_imgs[:int(len(train_imgs) * conf.ratio)]
        train_dataset = PCRLChestPretask(conf, train_imgs, transform=train_transform, train=True,
                                         spatial_transform=spatial_transform)
        print(len(train_dataset))
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_sampler = None
        dataloader = {}
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=conf.batch_size, shuffle=(train_sampler is None),
            num_workers=conf.workers, pin_memory=True, sampler=train_sampler)
        dataloader['train'] = train_loader
        dataloader['eval'] = train_loader
        return dataloader

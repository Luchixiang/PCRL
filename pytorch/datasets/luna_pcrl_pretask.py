import copy
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms


class PCRLLunaPretask(Dataset):
    def __init__(self, config, img_train, train=False, transform=None):
        self.config = config
        self.imgs = img_train
        self.train = train
        self.transform = transform
        self.norm = torchio.transforms.ZNormalization()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_name = self.imgs[index]
        pair = np.load(image_name)
        crop1 = pair[0]
        crop1 = np.expand_dims(crop1, axis=0)
        crop2 = pair[1]
        crop2 = np.expand_dims(crop2, axis=0)
        gt1 = copy.deepcopy(crop1)
        gt2 = copy.deepcopy(crop2)
        # gt1 = torch.tensor(gt1, dtype=torch.float)
        # gt2 = torch.tensor(gt2, dtype=torch.float)
        input1 = self.transform(crop1)
        input2 = self.transform(crop2)
        input1 = self.local_pixel_shuffling(input1, prob=self.config.local_rate)
        input2 = self.local_pixel_shuffling(input2, prob=self.config.local_rate)
        # input1 = self.nonlinear_transformation(input1, self.config.nonlinear_rate)
        # input2 = self.nonlinear_transformation(input2, self.config.nonlinear_rate)
        if random.random() < self.config.paint_rate:
            if random.random() < self.config.inpaint_rate:
                # Inpainting
                input1 = self.image_in_painting(input1)
                input2 = self.image_in_painting(input2)
            else:
                # Outpainting
                input1 = self.image_out_painting(input1)
                input2 = self.image_out_painting(input2)
        mask1 = copy.deepcopy(input1)
        mask2 = copy.deepcopy(input2)
        mask1, aug_tensor1 = self.spatial_aug(mask1)
        mask2, aug_tensor2 = self.spatial_aug(mask2)

        return torch.tensor(input1, dtype=torch.float), torch.tensor(input2, dtype=torch.float), \
               torch.tensor(mask1, dtype=torch.float), \
               torch.tensor(mask2, dtype=torch.float), \
               torch.tensor(gt1, dtype=torch.float), \
               torch.tensor(gt2, dtype=torch.float), aug_tensor1, aug_tensor2

    def spatial_aug(self, img):
        # img = img.numpy()
        aug_tensor = [0 for _ in range(7)]
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            aug_tensor[0] = 1
        if random.random() < 0.5:
            img = np.flip(img, axis=2)
            aug_tensor[1] = 1
        if random.random() < 0.5:
            img = np.flip(img, axis=3)
            aug_tensor[2] = 1
        times = int(random.random() // 0.25)
        img = np.rot90(img, k=times, axes=(1, 2))
        aug_tensor[times + 3] = 1
        return torch.tensor(img.copy(), dtype=torch.float), torch.tensor(aug_tensor)

    def bernstein_poly(self, i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """

        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.

           Control points should be a list of lists, or list of tuples
           such as [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000

            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals

    def data_augmentation(self, x, y, prob=0.5):
        # augmentation by flipping
        cnt = 3
        while random.random() < prob and cnt > 0:
            degree = random.choice([0, 1, 2])
            x = np.flip(x, axis=degree)
            y = np.flip(y, axis=degree)
            cnt = cnt - 1

        return x, y

    def nonlinear_transformation(self, x, prob=0.5):
        if random.random() >= prob:
            return x
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]
        xvals, yvals = self.bezier_curve(points, nTimes=100000)
        if random.random() < 0.5:
            # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        return nonlinear_x

    def local_pixel_shuffling(self, x, prob=0.5):
        if random.random() >= prob:
            return x
        image_temp = copy.deepcopy(x)
        orig_image = copy.deepcopy(x)
        _, img_rows, img_cols, img_deps = x.shape
        num_block = 10000
        for _ in range(num_block):
            block_noise_size_x = random.randint(1, img_rows // 10)
            block_noise_size_y = random.randint(1, img_cols // 10)
            block_noise_size_z = random.randint(1, img_deps // 10)
            noise_x = random.randint(0, img_rows - block_noise_size_x)
            noise_y = random.randint(0, img_cols - block_noise_size_y)
            noise_z = random.randint(0, img_deps - block_noise_size_z)
            window = orig_image[0, noise_x:noise_x + block_noise_size_x,
                     noise_y:noise_y + block_noise_size_y,
                     noise_z:noise_z + block_noise_size_z,
                     ]
            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_noise_size_x,
                                     block_noise_size_y,
                                     block_noise_size_z))
            image_temp[0, noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = window
        local_shuffling_x = image_temp

        return local_shuffling_x

    def image_in_painting(self, x):
        _, img_rows, img_cols, img_deps = x.shape
        cnt = 5
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
            block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
            block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y,
                                                                   block_noise_size_z, ) * 1.0
            cnt -= 1
        return x

    def image_out_painting(self, x):
        _, img_rows, img_cols, img_deps = x.shape
        image_temp = copy.deepcopy(x)
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        cnt = 4
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
            block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
            block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                    noise_y:noise_y + block_noise_size_y,
                                                    noise_z:noise_z + block_noise_size_z]
            cnt -= 1
        return x

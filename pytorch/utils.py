import os
import torch
import numpy as np
from PIL import ImageFilter
import random
import math


def get_chest_list(txt_path, data_dir):
    image_names = []
    labels = []
    with open(txt_path, "r") as f:
        for line in f:
            items = line.split()
            image_name = items[0]
            label = items[1:]
            label = [int(i) for i in label]
            image_name = os.path.join(data_dir, image_name)
            image_names.append(image_name)
            labels.append(label)
    return image_names, labels


def get_luna_list(config, train_fold, valid_fold, test_fold, suffix, file_list):
    x_train = []
    x_valid = []
    x_test = []
    for i in train_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                if file_list is not None and file.split('_')[0] in file_list:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
                elif file_list is None:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
    for i in valid_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                x_valid.append(os.path.join(config.data, 'subset' + str(i), file))
    for i in test_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                x_test.append(os.path.join(config.data, 'subset' + str(i), file))
    return x_train, x_valid, x_test


def get_luna_pretrain_list(ratio, path, train_fold):
    x_train = []
    # for i in train_fold:
    #     for file in os.listdir(os.path.join(path, 'subset' + str(i))):
    #         if 'mhd' in file:
    #             x_train.append(file[:-4])
    with open('train_val_txt/luna_train.txt', 'r') as f:
        for line in f:
            x_train.append(line.strip('\n'))
    return x_train[:int(len(x_train) * ratio)]


def get_luna_finetune_list(ratio, path, train_fold):
    x_train = []
    with open('train_val_txt/luna_train.txt', 'r') as f:
        for line in f:
            x_train.append(line.strip('\n'))
    return x_train[int(len(x_train) * ratio):]


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def adjust_learning_rate(epoch, args, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs_list = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs_list.append(int(it))
    # steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs_list))
    # if steps > 0:
    #     new_lr = opt.lr * (opt.lr_decay_rate ** steps)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = new_lr
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_params(output_size, img_size=(224, 224)):
    h, w = img_size
    th, tw = output_size
    i = random.randint(0, h - th + 1)
    j = random.randint(0, w - tw + 1)
    return i, j, th, tw


def get_random_2dboxs(box_num, batch_size):
    """
        get the random box on batch images
     Arguments:
         box_num: the number of boxes on every image
         batch_size: batch size
     Returns:
         a set of boxes : [K, 5]
         K is batch_size * box_num
         every box correspond to [batch_index, x1,y1,x2,y2]
         batch_index is the index of image in the batch
     """
    boxes = []
    for bid in range(batch_size):
        bboxes = []
        while len(bboxes) < box_num:
            h = random.randint(50, 223)  # 下限调高一点
            w = random.randint(50, 223)
            if 10000 <= h * w <= 30000:
                i, j, h, w = get_params((h, w))
                bboxes.append((i, j, i + h, j + w))
                boxes.append((bid, i, j, i + h, w + j))
    return np.array(boxes)


def get_batch_random_crop(box_num=50, batch_size=128):
    boxes = []
    all_boxes = []
    while len(boxes) < box_num:
        h = random.randint(50, 223)  # 下限调高一点
        w = random.randint(50, 223)
        if 10000 <= h * w <= 30000:
            i, j, h, w = get_params((h, w))
            # bboxes.append((i, j, i + h, j + w))
            boxes.append((i, j, i + h, w + j))
    for bid in range(batch_size):
        for box in boxes:
            i, j, m, n = box
            all_boxes.append((bid, i, j, m, n))
    return np.array(boxes), np.array(all_boxes)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    # pdb.set_trace()
    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def same(boxes):
    """
     return a matrix whose element denotes whether the element
     in the boxes is in the same image
     Arguments:
         boxes (Tensor[N]): batch_index of boxes
     Returns:
         area (Tensor[N, N])
     """
    boxes = boxes.unsqueeze(dim=1)
    boxes2 = boxes.t()
    return torch.eq(boxes, boxes2)

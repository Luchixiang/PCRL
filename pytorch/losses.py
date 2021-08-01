import torch.nn.functional as F
import torch
import torch.nn as nn


class Multi_Soft_Dice_Loss(nn.Module):
    def __init__(self):
        super(Multi_Soft_Dice_Loss, self).__init__()

    def forward(self, result, target, total_classes=5, train=True):
        loss = 0.0
        # print(result.shape, target.shape)
        target = target[:, 0]
        # softmax = nn.Softmax(dim = 1)
        # result = softmax(result)
        for i in range(result.size(0)):
            epsilon = 1e-6
            Loss = []
            weight = [2, 0.4, 0.9, 0.7]
            for j in range(1, total_classes):
                result_sum = torch.sum(result[i, j, :, :, :])
                target_sum = torch.sum(target[i, :, :, :] == j)
                # print(target_sum.data)
                intersect = torch.sum(result[i, j, :, :, :] * (target[i, :, :, :] == j).float())
                # print(result_sum, target_sum, intersect)
                dice = (2. * intersect + epsilon) / (target_sum + result_sum + epsilon)
                # print("The {} batch's {} class's dice is {}".format(i, j, dice))
                Loss.append(1 - dice)
            for m in range(4):
                if train:
                    loss += Loss[m] * weight[m]
                else:
                    loss += Loss[m]
        return loss / result.size(0)


def bceDiceLoss(input, target, train=True):
    bce = F.binary_cross_entropy_with_logits(input, target)
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    if train:
        return dice + 0.2 * bce
    return dice


def thor_dice_loss(input, target, train=True):
    # print(input.shape, target.shape)
    es_dice = bceDiceLoss(input[:, 0], target[:, 0], train)
    tra_dice = bceDiceLoss(input[:, 1], target[:, 1], train)
    aor_dice = bceDiceLoss(input[:, 2], target[:, 2], train)
    heart_dice = bceDiceLoss(input[:, 3], target[:, 3], train)
    print(f'label1 dice {es_dice}, label2 dice {tra_dice}, label3 dice{aor_dice}, label4 dice{heart_dice}')
    return es_dice + tra_dice + aor_dice + heart_dice


def dice_loss(input, target, train=True):
    wt_loss = bceDiceLoss(input[:, 0], target[:, 0], train)
    tc_loss = bceDiceLoss(input[:, 1], target[:, 1], train)
    et_loss = bceDiceLoss(input[:, 2], target[:, 2], train)
    print(f'wt loss: {wt_loss}, tc_loss : {tc_loss}, et_loss: {et_loss}')
    return wt_loss + tc_loss + et_loss


class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1
        eps = 1e-5

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class NCEKLLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCEKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        x = F.log_softmax(x, dim=1)
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss

"""
Training code for C2L
"""
from __future__ import print_function

import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from models.pcrl_model_3d import PCRLModel3d
from losses import NCECriterion
from memory_c2l import MemoryC2L
from models import InsResNet18
from utils import adjust_learning_rate, AverageMeter
import segmentation_models_pytorch as smp

try:
    from apex import amp, optimizers
except ImportError:
    pass


def Normalize(x):
    norm_x = x.pow(2).sum(1, keepdim=True).pow(1. / 2.)
    x = x.div(norm_x)
    return x


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def mixup_data(x, y, alpha=1.0, index=None, lam=None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if lam is None:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = lam

    lam = max(lam, 1 - lam)
    batch_size = x.size()[0]
    if index is None:
        index = torch.randperm(batch_size).cuda()
    else:
        index = index

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam, index


def train_pcrl_3d(args, data_loader, out_channel=3):
    torch.autograd.set_detect_anomaly(True)
    nce_k = 16384
    nce_t = args.moco_t
    nce_m = 0.5
    train_loader = data_loader['train']
    # create model and optimizer
    n_data = len(train_loader)
    model = PCRLModel3d(in_channels=1, n_class=1, student=True, norm=args.norm)
    model_ema = PCRLModel3d(in_channels=1, n_class=1, student=False, norm=args.norm)
    model = model.cuda()
    model_ema = model_ema.cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    optimizer_ema = torch.optim.SGD(model_ema.parameters(),
                                    lr=0,
                                    momentum=0,
                                    weight_decay=0)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model_ema, optimizer_ema = amp.initialize(model_ema, optimizer_ema, opt_level='O1')
    # set the contrast memory and
    contrast = MemoryC2L(128, n_data, nce_k, nce_t, False).cuda()
    criterion = NCECriterion(n_data).cuda()
    criterion2 = nn.MSELoss().cuda()

    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)

    moment_update(model, model_ema, 0)

    for epoch in range(0, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        loss, prob = train_rep_C2L(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, criterion2,
                                   )
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # save model
        if epoch % 60 == 0:
            # saving the model
            print('==> Saving...')
            state = {'opt': args, 'state_dict': model.module.state_dict(),
                     'contrast': contrast.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch, 'model_ema': model_ema.state_dict()}

            save_file = os.path.join(args.output,
                                     args.model + "_" + args.n + '_' + args.phase + '_' + str(
                                         args.ratio) + '_' + str(epoch) + '.pt')
            torch.save(state, save_file)
            # help release GPU memory
            del state
        if epoch == 242:
            break

        torch.cuda.empty_cache()


def train_rep_C2L(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, criterion2):
    """
    one epoch training for instance discrimination
    """

    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    c2l_loss_meter = AverageMeter()
    mg_loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (input1, input2, mask1, mask2, gt1, gt2, aug_tensor1, aug_tensor2) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = input1.size(0)
        x1 = input1.float().cuda()
        x2 = input2.float().cuda()
        mask1 = mask1.float().cuda()
        mask2 = mask2.float().cuda()
        aug_tensor1 = aug_tensor1.float().cuda()
        aug_tensor2 = aug_tensor2.float().cuda()
        gt1 = gt1.float().cuda()
        gt2 = gt2.float().cuda()
        # ===================forward=====================
        # ids for ShuffleBN
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)
        alpha1 = np.random.beta(1., 1.)
        alpha1 = max(alpha1, 1 - alpha1)
        with torch.no_grad():
            x2 = x2[shuffle_ids]
            feat_k, feats_k = model_ema(x2)
            feats_k = [tmp[reverse_ids] for tmp in feats_k]
            feat_k = feat_k[reverse_ids]
            x2 = x2[reverse_ids]
        feat_q, unet_out_alpha, unet_out = model(x1, feats_k, alpha1, aug_tensor1, aug_tensor2)
        out = contrast(Normalize(feat_q), Normalize(feat_k))
        mixed_x1, mixed_feat1, lam1, index = mixup_data(x1.clone(),
                                                        feat_q.clone())
        mixed_x2, mixed_feat2, _, _ = mixup_data(x2.clone(), feat_k.clone(),
                                                 index=index, lam=lam1)
        mixed_gt1, _, _, _ = mixup_data(gt1.clone(), feat_q.clone(), index=index, lam=lam1)
        mixed_gt2, _, _, _ = mixup_data(gt2.clone(), feat_q.clone(), index=index, lam=lam1)
        alpha2 = np.random.beta(1., 1.)
        alpha2 = max(alpha2, 1 - alpha2)
        with torch.no_grad():
            mixed_x2 = mixed_x2[shuffle_ids]
            mixed_feat_k, mixed_feats_k = model_ema(mixed_x2)
            mixed_feats_k = [tmp[reverse_ids] for tmp in mixed_feats_k]
            mixed_feat_k = mixed_feat_k[reverse_ids]
            mixed_x2 = mixed_x2[reverse_ids]

        mixed_feat_q, mixed_unet_out_alpha, mixed_unet_out = model(mixed_x1, mixed_feats_k, alpha2, aug_tensor1,
                                                                   aug_tensor2, mixup=True)
        mixed_feat_q_norm = Normalize(mixed_feat_q)
        mixed_feat_k_norm = Normalize(mixed_feat_k)
        mixed_feat1_norm = Normalize(mixed_feat1)
        mixed_feat2_norm = Normalize(mixed_feat2)

        out2 = contrast(mixed_feat_q_norm, mixed_feat_k_norm)
        out3 = contrast(mixed_feat_q_norm, mixed_feat2_norm)
        c2l_loss = (criterion(out) + criterion(out2) + criterion(out3)) / 3.
        # c2l_loss = criterion(out)
        c2l_loss_meter.update(c2l_loss.item(), bsz)
        mg_loss = (criterion2(unet_out, mask1) + criterion2(mixed_unet_out, mixed_gt1)
                   + criterion2(unet_out_alpha, alpha1 * mask1 + (1 - alpha1) * mask2) +
                   criterion2(mixed_unet_out_alpha, alpha2 * mixed_gt1 + (1 - alpha2) * mixed_gt2)) / 4.
        # mg_loss = (criterion2(unet_out, mask1) + criterion2(unet_out_alpha,
        # alpha1 * mask1 + (1 - alpha1) * mask2)) / 2.
        loss = c2l_loss + mg_loss
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # ===================meters=====================
        mg_loss_meter.update(mg_loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        moment_update(model, model_ema, 0.999)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 5 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'c2l loss {c2l_loss.val:.3f} ({c2l_loss.avg:.3f})\t'
                  'mg loss {mg_loss.val:.3f} ({mg_loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, c2l_loss=c2l_loss_meter, mg_loss=mg_loss_meter, prob=prob_meter))
            print(out.shape)
            sys.stdout.flush()

    return mg_loss_meter.avg, prob_meter.avg

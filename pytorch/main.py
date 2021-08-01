import argparse
import os
import warnings

import torch.backends.cudnn

from initiater import *

import torch.multiprocessing as mp

from pcrl import train_pcrl_2d
import sys
from pcrl_3d import train_pcrl_3d

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--data', metavar='DIR', default='/data1/luchixiang/LUNA16/processed',
                        help='path to dataset')
    parser.add_argument('--model', metavar='MODEL', default='model_genesis', help='choose the model')
    parser.add_argument('--phase', default='pretask', type=str, help='pretask or finetune or train from scratch')
    parser.add_argument('--b', default=16, type=int, help='batch size')
    parser.add_argument('--weight', default=None, type=str, help='weight to load')
    parser.add_argument('--epochs', default=100, type=int, help='epochs to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output', default='./model_genesis_pretrain', type=str, help='output path')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer to use')
    parser.add_argument('--outchannel', default=1, type=int, help='classes num')
    parser.add_argument('--n', default='luna', type=str, help='dataset to use')
    parser.add_argument('--d', default=3, type=int, help='3d or 2d to run')
    parser.add_argument('--workers', default=4, type=int, help='num of workers')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='gpu indexs')
    parser.add_argument('--loss', default='mse', type=str, help='loss to use')
    parser.add_argument('--patience', default=50, type=int, help='patience')
    parser.add_argument('--inchannel', default=1, type=int, help='input channels')
    parser.add_argument('--ratio', default=0.8, type=float, help='ratio of data used for pretraining')
    parser.add_argument('--root', default='/data1/luchixiang/LUNA16', help='root path')
    parser.add_argument('--step', default=50, type=int)
    parser.add_argument('--norm', default='bn')
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=1e-4)
    parser.add_argument('--final_act', default='sigmoid')
    parser.add_argument('--lr_decay_epochs', type=str, default='160,200,240,280,320',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--moco_t', default=0.2, type=float)

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # torch.backends.cudnn.benchmark = True
    conf = get_config(args)
    data_loader = get_dataloader(args, conf)
    criterion = get_loss(args)
    train_generator = data_loader['train']
    valid_generator = data_loader['eval']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count())
    intial_epoch = 0
    sys.stdout.flush()
    if args.model == 'pcrl' and args.phase == 'pretask' and args.d == 2:
        train_pcrl_2d(args, data_loader)
    elif args.model == 'pcrl' and args.phase == 'pretask' and args.d == 3:
        train_pcrl_3d(args, data_loader)

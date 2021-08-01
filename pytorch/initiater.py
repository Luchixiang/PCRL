from config import models_genesis_config
from data import DataGenerator
from losses import *
from models import *
import torch.nn as nn
import torch.nn.init as init


def get_config(args):
    conf = models_genesis_config()
    conf.batch_size = args.b
    conf.lr = args.lr
    conf.weights = args.weight
    conf.model_path = args.output
    conf.nb_epoch = args.epochs
    conf.data = args.data
    conf.optimizer = args.optimizer
    conf.workers = args.workers
    conf.nb_class = args.outchannel
    conf.patience = args.patience
    conf.ratio = args.ratio
    conf.root = args.root
    return conf


def get_dataloader(args, conf):
    generator = DataGenerator(conf)
    loader_name = args.model + '_' + args.n + '_' + args.phase
    print(loader_name)
    dataloader = getattr(generator, loader_name)()
    return dataloader



def kaiming_normal(net, a=0, mode='fan_in', nonlinearity='relu'):
    for m in net.modules():
        if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)):
            init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if m.bias is not None:
                init.constant_(m.bias, 0.)
        else:
            pass
    return net


def get_loss(args):
    if args.loss == 'gan':
        print('using gan loss')
        criterion = [nn.L1Loss(), nn.BCELoss()]
    elif args.loss == 'mse':
        print('using mse loss')
        criterion = nn.MSELoss()
    elif args.loss == 'dice':
        print('using dice loss')
        criterion = dice_loss
    elif args.loss == 'thor_dice':
        # weights = torch.FloatTensor([0.5, 1.0, 1.0, 1.0, 1.0]).cuda()
        # criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
        criterion = Multi_Soft_Dice_Loss()
    elif args.loss == 'bce':
        print('using bce loss')
        criterion = nn.BCELoss()
    elif args.loss == 'semantic':
        print('using semantic loss')
        criterion = [nn.MSELoss().cuda(), nn.CrossEntropyLoss().cuda()]
    elif args.loss == 'softmax':
        print('using softmax loss')
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'nce':
        print('using nce loss')
        criterion = NCECriterion(10)
    return criterion

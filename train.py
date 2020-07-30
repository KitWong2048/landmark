import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import argparse



import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tqdm import tqdm

import debug
from src import qsub
from src import utils, data_utils, metrics
from src.eval_retrieval import eval_datasets
from src.modeling import batch_norm, models



parser = argparse.ArgumentParser()
parser.add_argument('--devices', '-d', type=str, default="0" help='comma delimited gpu device list (e.g. "0,1")')
parser.add_argument('--resume', type=str, default=None, help='checkpoint path')
parser.add_argument('--save-interval', '-s', type=int, default=1, help='if 0 or negative value, not saving')
args = parser.parse_args()



#ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
ROOT = '../' 

params = {
    #'ex_name': __file__.replace('.py', ''),
    'ex_name':'experiment1'
    'seed': 123456789,
    'lr': 1e-3,
    'batch_size': 32,
    'test_batch_size': 64,
    'optimizer': 'momentum',
    'epochs': 5,
    'image_size': 512,
    'scaleup_epochs': 0,
    'wd': 1e-5,
    'model_name': 'resnet101',
    'pooling': 'GeM',
    'use_fc': True,
    'loss': 'arcface',
    'margin': 0.3,
    's': 30,
    'theta_zero': 1.25,
    'fc_dim': 512,
    'scale_limit': 0.2,
    'shear_limit': 0,
    'brightness_limit': 0.0,
    'contrast_limit': 0.0,
    'augmentation': 'soft',
    'train_data': 'gld_v2',
    'freeze_bn': True,
    'verifythresh': 30,
    'freqthresh': 3,
    #'base_ckpt_path': 'exp/v1only/ep4_augmentation-soft_epochs-5_loss-arcface.pth',
    'data_root': ROOT+os.sep+'landmark-retrieval-2020',
    'train_csv': ROOT+os.sep+'landmark-retrieval-2020/train.csv'，
    'val_csv': ROOT+os.sep+'landmark-retrieval-2020/val.csv'
}





if not Path(ROOT + f'exp/{params["ex_name"]}').exists():
    Path(ROOT + f'exp/{params["ex_name"]}').mkdir(parents=True)

if not Path(ROOT + f'exp/{params["ex_name"]}/check').exists():
    Path(ROOT + f'exp/{params["ex_name"]}/check').mkdir(parents=True)

if not Path(ROOT + f'exp/{params["ex_name"]}/log').exists():
    Path(ROOT + f'exp/{params["ex_name"]}/log').mkdir(parents=True)

np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
torch.cuda.manual_seed_all(params['seed'])
torch.backends.cudnn.benchmark = False


def job(devices, resume, save_interval):
    global params

    mode_str = 'train'
    setting = ''

    exp_path = ROOT + f'exp/{params["ex_name"]}/'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

    logger, writer = utils.get_logger(log_dir=exp_path + f'{mode_str}/log/{setting}',
                                      tensorboard_dir=exp_path + f'{mode_str}/tf_board/{setting}')
    log_file = open(ROOT + f'exp/{params["ex_name"]}/log', 'a+')


    if params['augmentation'] == 'soft':
        params['scale_limit'] = 0.2
        params['brightness_limit'] = 0.1
    elif params['augmentation'] == 'middle':
        params['scale_limit'] = 0.3
        params['shear_limit'] = 4
        params['brightness_limit'] = 0.1
        params['contrast_limit'] = 0.1
    else:
        raise ValueError

    train_transform, eval_transform = data_utils.build_transforms(
        scale_limit=params['scale_limit'],
        shear_limit=params['shear_limit'],
        brightness_limit=params['brightness_limit'],
        contrast_limit=params['contrast_limit'],
    )



    data_loaders = data_utils. make_data_loaders(data_root=params['data_root'],
                       train_csv=params['train_csv'],
                       val_csv=params['val_csv'],
                       train_transform=train_transform,
                       eval_transform=eval_transform,
                       size = (paras['image_size'],paras['image_size']),
                       batch_size = params['batch_size']，
                       num_workers=8):



    model = models.LandmarkNet(n_classes=params['class_topk'],
                               model_name=params['model_name'],
                               pooling=params['pooling'],
                               loss_module=params['loss'],
                               s=params['s'],
                               margin=params['margin'],
                               theta_zero=params['theta_zero'],
                               use_fc=params['use_fc'],
                               fc_dim=params['fc_dim'],
                               )
    num_GPU = len(devices.split(',')) > 1
    in num_GPU>0:
        model = model.cuda() 

    criterion = nn.CrossEntropyLoss()
    optimizer = utils.get_optim(params, model)

    if resume:
        #sdict = torch.load(ROOT + params['base_ckpt_path'])['state_dict']
        #del sdict['final.weight']  # remove fully-connected layer
        #model.load_state_dict(sdict, strict=False)
        rets = load_checkpoint(resume, model=model, optimizer=None, params=False, epoach=True)
    


    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params['epochs'] * len(data_loaders['train']), eta_min=3e-6)
    start_epoch, end_epoch = (0, params['epochs'] - params['scaleup_epochs'])
    start_epoch = rets['epoch']

    for _ in range(start_epoch*len(data_loaders['train'])):
        scheduler.step()


    if len(devices.split(',')) > 1:
        model = nn.DataParallel(model)

    for epoch in range(start_epoch, end_epoch):
    #while epoch <= end_epoch:
        logger.info(f'Epoch {epoch}/{end_epoch}')

        # ============================== train ============================== #
        model.train(True)

        losses = utils.AverageMeter()
        prec1 = utils.AverageMeter()

        for i, (_, x, y) in tqdm(enumerate(data_loaders['train']),
                                 total=len(data_loaders['train']),
                                 miniters=None, ncols=55):
            
            if num_GPU>0:
                x = x.to('cuda')
                y = y.to('cuda')

            outputs = model(x, y)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            acc = metrics.accuracy(outputs, y)
            losses.update(loss.item(), x.size(0))
            prec1.update(acc, x.size(0))

            if i % 100 == 99:
                logger.info(f'{epoch+i/len(data_loaders["train"]):.2f}epoch | {setting} acc: {prec1.avg}')

        train_loss = losses.avg
        train_acc = prec1.avg

        print("[{:5d}] => loss={:.9f}, acc={:.9f}; lr={:.9f}.".format(epoch, train_loss, train_acc))


        writer.add_scalars('Loss', {'train': train_loss}, epoch)
        writer.add_scalars('Acc', {'train': train_acc}, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) == end_epoch or (epoch + 1) % save_interval == 0:
            output_file_name = exp_path + f'ep{epoch}_' + setting + '.pth'
            utils.save_checkpoint(path=output_file_name,
                                  model=model,
                                  epoch=epoch,
                                  optimizer=optimizer,
                                  params=params)


        # ============================== validation ============================== #
        model.eval()
        val_losses = utils.AverageMeter()
        val_prec1 = utils.AverageMeter()

        with torch.no_grad():
            for i, (_, x, y) in tqdm(enumerate(data_loaders['val']),
                                 total=len(data_loaders['val']),
                                 miniters=None, ncols=55):

                if num_GPU>0:
                    x = x.to('cuda')
                    y = y.to('cuda')

                outputs = model(x, y)
                loss = criterion(outputs, y)

                acc = metrics.accuracy(outputs, y)
                val_losses.update(loss.item(), x.size(0))
                val_prec1.update(acc, x.size(0))
            
        val_loss = val_losses.avg
        val_acc = val_prec1.avg
        print("[{:5d}] => loss={:.9f}, acc={:.9f}; lr={:.9f}.".format(epoch, val_loss, val_acc))
        log_file.write("[{:5d}] => loss={:.9f}, acc={:.9f}; lr={:.9f}.".format(epoch, val_loss, val_acc))
        log_file.write("\n")

    log_file.close()

if __name__ == '__main__':
    job(args.devices, args.resume, args.save_interval)

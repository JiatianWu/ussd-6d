from __future__ import print_function
import argparse
import os
import shutil
import time
import sys
import random
import math

import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib as mpl
import matplotlib.pyplot as plt

import logger
import utils
from cfg import parse_cfg
from darknet import Darknet
# import _init_paths
# from datasets.factory import get_imdb
# from custom import *
from dataloader import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--arch', default='localizer_alexnet')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model', default=True)
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--vis',action='store_true')

use_cuda      = False

if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

cfgfile       = 'cfg/ussd.cfg'

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train(epoch):
    global processed_batches
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    # train_loader = torch.utils.data.DataLoader(
    #     dataset.listDataset(trainlist, shape=(init_width, init_height),
    #                    shuffle=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                    ]), 
    #                    train=True, 
    #                    seen=cur_model.seen,
    #                    batch_size=batch_size,
    #                    num_workers=num_workers),
    #     batch_size=batch_size, shuffle=False, **kwargs)
    
    batch_size = 1
    num_train_class = 10
    total_image, total_grid = total_image_loader(num_train_class)
    total_image = np.array(total_image)
    total_image = Variable(torch.FloatTensor(total_image))
    total_grid = np.array(total_grid)
    total_grid = Variable(torch.FloatTensor(total_grid))

    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)

    num_batch = num_train_class * 1313 / batch_size
    idx = 0
    for batch_idx in num_batch:
    # for batch_idx, (data, target) in enumerate(train_loader):
        data_tmp = total_image[idx : idx + 3 * batch_size, :, :, :]
        data = data_tmp.view(batch_size,  3 * data_tmp.shape[1], data_tmp.shape[2], data_tmp.shape[3])

        target_tmp = total_grid[idx : idx + 3 * batch_size, :, :, :]
        target = target_tmp.view(batch_size, 3, target_tmp.shape[1], target_tmp.shape[2], target_tmp.shape[3])   


        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        if use_cuda:
            data = data.cuda()
            #target= target.cuda()
        t3 = time.time()
        data, target = Variable(data), Variable(target)
        t4 = time.time()
        optimizer.zero_grad()
        t5 = time.time()
        output = model(data)
        t6 = time.time()
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss = region_loss(output, target)
        t7 = time.time()
        loss.backward()
        t8 = time.time()
        optimizer.step()
        t9 = time.time()
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
    print('')
    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    if (epoch+1) % save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    model = Darknet(cfgfile)
    model.print_network()
    model.load_weights('cfg/yolo.weights')

    region_loss = model.loss

    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
    # if args.arch=='localizer_alexnet':
    #     model = localizer_alexnet(pretrained=args.pretrained)
    # elif args.arch=='localizer_alexnet_robust':
    #     model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    # model.features = torch.nn.DataParallel(model.features)
    # model.cuda()

    # # TODO:
    # # define loss function (criterion) and optimizer
    # criterion = nn.BCELoss().cuda()
    # optimizer = torch.optim.SGD(model.classifier.parameters(), args.lr, 
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # # set random seed
    # torch.manual_seed(512)
    # np.random.seed(512)

    # # Data loading code
    # # TODO: Write code for IMDBDataset in custom.py
    # trainval_imdb = get_imdb('voc_2007_trainval')
    # test_imdb = get_imdb('voc_2007_test')

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # train_dataset = IMDBDataset(
    #     trainval_imdb,
    #     transforms.Compose([
    #         transforms.Resize((512,512)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # train_sampler = None
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     IMDBDataset(test_imdb, transforms.Compose([
    #         transforms.Resize((384,384)),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # if args.evaluate:
    #     validate(val_loader, model, criterion)
    #     return

    # # TODO: Create loggers for visdom and tboard
    # # TODO: You can pass the logger objects to train(), make appropriate
    # # modifications to train()
    # if args.arch == 'localizer_alexnet':
    #     data_log = logger.Logger('./logs/', name='freeloc')
    #     vis = visdom.Visdom(server='http://localhost', port='8097')
    #     args.epochs = 32
    # else:
    #     data_log = logger.Logger('./log_robust', name='freeloc')
    #     vis = visdom.Visdom(server='http://localhost', port='8090')
    #     args.epochs = 47


    # for epoch in range(args.start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, epoch)

    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch, data_log, vis)

    #     # evaluate on validation set
    #     if epoch%args.eval_freq==0 or epoch==args.epochs-1:
    #         m1, m2 = validate(val_loader, model, criterion, epoch, data_log, vis)
    #         score = m1*m2
    #         # remember best prec@1 and save checkpoint
    #         is_best =  score > best_prec1
    #         best_prec1 = max(score, best_prec1)
    #         save_checkpoint({
    #             'epoch': epoch + 1,
    #             'arch': args.arch,
    #             'state_dict': model.state_dict(),
    #             'best_prec1': best_prec1,
    #             'optimizer' : optimizer.state_dict(),
    #         }, is_best)

# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    # MAP
    batch_size = target.size(0)
    class_types = target.size(1)
    output_cpu = output.cpu().numpy().astype('float32')
    target_cpu = target.cpu().numpy().astype('float32')
    res = 0.0
    for i in range(batch_size):
        pred_cls = output_cpu[i]
        gt_cls = target_cpu[i]
        
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=None)
        res += ap

    return [res/batch_size]

def metric2(output, target):
    # TODO: Ignore for now - proceed till instructed
    # Top 3 accuracy 
    res = 0.0
    for i in range(target.shape[0]):
        class_i = output[i]
        sort_idx = sorted(range(len(class_i)), key=lambda j: class_i[j])
        output_top3 = np.take(output[i], sort_idx[-3:])
        target_top3 = np.take(target[i], sort_idx[-3:])
        res= res+int((output_top3*target_top3).sum()!=0)*1.0

    return [res/target.shape[0]]

if __name__ == '__main__':
    main()
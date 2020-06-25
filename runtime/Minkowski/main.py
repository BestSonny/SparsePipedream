#first import dataset and open3d before import torch to avoid error in docker
import dataset.dataset as dataset
import torch.utils.data as data
import os
import os.path
import numpy as np
import logging
import sys
import json
import argparse
from models.minkunet import MinkUNet34C
import models.vgg16.vgg_mink as vgg
import time
import MinkowskiEngine as ME
import random
import importlib
from multiprocessing import Manager
import torch
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(0)
import numpy as np
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from apex import amp

parser = argparse.ArgumentParser(description='PyTorch Minkowski Training')
parser.add_argument('--data_dir', type=str, required=True, help="dataset path")
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains full model definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--voxel_size', type=float, default=0.05)
parser.add_argument('--max_iter', type=int, default=120000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--dataset', default='modelnet40', type=str,
                    help='dataset name, modelnet40 or shapenet')

best_prec1 = 0
args = parser.parse_args()

# initialize Amp
amp_handle = amp.init(enabled=args.fp16)

class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length

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

def main():
    global args, best_prec1

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    model = module.full_model()
    #model = vgg.vgg16_bn(out_channels=40)
    print("model:", model)

    model = model.cuda()
    #if not args.distributed:
    #    model = torch.nn.DataParallel(model).cuda()
    #else:
    #    model.cuda()
    #    model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    global model_parameters, master_parameters
    #optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             betas=(0.9,0.999),
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    # Data loading code
    manager = Manager()
    shared_dict = manager.dict() #create cache for storing data
    if args.dataset == 'shapenet':
        train_dataset = dataset.ShapeNetDataset(root=args.data_dir,
                                                shared_dict=shared_dict,
                                                classification=True,
                                                voxel_size=args.voxel_size)
        val_dataset = dataset.ShapeNetDataset(root=args.data_dir,
                                              classification=True,
                                              split='val',
                                              voxel_size=args.voxel_size)
    else:
        train_transpose = dataset.Compose([dataset.RandomRotation(axis=np.array([0, 0, 1])),
                                           dataset.RandomTranslation(),
                                           dataset.RandomScale(0.8, 1.2),
                                           dataset.RandomShear()])
        train_dataset = dataset.ModelNet40Dataset(root=args.data_dir,
                                                  shared_dict=shared_dict,
                                                  split='train',
                                                  voxel_size=args.voxel_size,
                                                  transform=train_transpose)
        val_dataset = dataset.ModelNet40Dataset(root=args.data_dir,
                                                shared_dict={},
                                                split='val',
                                                voxel_size=args.voxel_size)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler  = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=(train_sampler is None),
                                             num_workers=args.workers,
                                             pin_memory=True, sampler=train_sampler,
                                             collate_fn=dataset.collate_pointcloud_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.eval_batch_size, 
                                             shuffle=False,
                                             num_workers=args.workers, pin_memory=True,
                                             sampler=val_sampler,
                                             collate_fn=dataset.collate_pointcloud_fn)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch) 
        adjust_learning_rate(optimizer, epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(train_loader, model, criterion, epoch) 
        else:
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            #checkpoint_dict = {
            #    'epoch': epoch + 1,
            #    'arch': args.arch,
            #    'state_dict': model.state_dict(),
            #    'best_prec1': best_prec1,
            #    'optimizer' : optimizer.state_dict(),
            #}
            # save_checkpoint(checkpoint_dict, is_best)
            print("Epoch: %d, best_prec1: %f" % (epoch + 1, best_prec1))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    epoch_start_time = time.time()

    for i, data_dict in enumerate(train_loader):
        if args.num_minibatches is not None and i >= args.num_minibatches:
            break
        coords = data_dict['coords']
        feats = data_dict['feats']
        labels = data_dict['labels']
        sin = ME.SparseTensor(
            feats, #coords[:, :3] * args.voxel_size,
            coords.int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        )  #.to(device)
        sin = sin.to('cuda')
        labels = labels.to('cuda')
        torch.cuda.synchronize()
        data_time.update(time.time() - end)

        sout = model(sin)
        loss = criterion(sout.F, labels)
        # measure accuracy and record loss
        if isinstance(sout, tuple):
            prec1, prec5 = accuracy(sout[0].F, labels, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(sout.F, labels, topk=(1, 5))
        losses.update(loss.item(), args.batch_size)
        top1.update(prec1[0], args.batch_size)
        top5.update(prec5[0], args.batch_size)

        #torch.cuda.synchronize() 
        #forward_time.update(time() - st3)

        #st4 = time()
        optimizer.zero_grad()
        with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            n = len(train_loader)
            if args.num_minibatches is not None:
                n = args.num_minibatches
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, n, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5,
                   memory=(float(torch.cuda.memory_allocated()) / 10**9),
                   cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
            import sys; sys.stdout.flush()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    epoch_start_time = time.time()
    with torch.no_grad():
        end = time.time()
        for i, data_dict in enumerate(val_loader):
            if args.num_minibatches is not None and i >= args.num_minibatches:
                break
            coords = data_dict['coords'] 
            feats = data_dict['feats'] 
            labels = data_dict['labels']
            sin = ME.SparseTensor(
                feats, #coords[:, :3] * args.voxel_size,
                coords.int(),
                allow_duplicate_coords=True,  # for classification, it doesn't matter
            )  #.to(device)
            sin = sin.to('cuda')
            labels = labels.to('cuda')
            # compute output
            sout = model(sin)
            loss = criterion(sout.F, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(sout.F, labels, topk=(1, 5))
            losses.update(loss.item(), args.eval_batch_size)
            top1.update(prec1[0], args.eval_batch_size)
            top5.update(prec5[0], args.eval_batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                n = len(val_loader)
                if args.num_minibatches is not None:
                    n = args.num_minibatches
                print('Test: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, n, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5,
                       memory=(float(torch.cuda.memory_allocated()) / 10**9),
                       cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                import sys; sys.stdout.flush()

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


if __name__ == '__main__':
    #config = parser.parse_args()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #net = MinkUNet34C(3, 16, D=3)
    #net.to(device)
    #train(net, device, config)
    main()

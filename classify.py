import argparse
import shutil
import time

import numpy as np
import os

import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import dla
import dataset

from folder import ImageFolder
import data_transforms

model_names = sorted(name for name in dla.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(dla.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='DLA ImageNet Training')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--data-name', default='imagenet',
                        help='Name of the dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='dla34',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: dla34)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--check-freq', default=1, type=int,
                         help='print frequency (default: 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                        help='use pre-trained model for '
                             'the specified dataset.')
    parser.add_argument('--classes', default=1000, type=int,
                        help='Number of classes in the model')
    parser.add_argument('--lr-adjust', dest='lr_adjust',
                        choices=['step'], default='step')
    parser.add_argument('--crop-size', dest='crop_size', type=int, default=224)
    parser.add_argument('--scale-size', dest='scale_size', type=int,
                        default=256)
    parser.add_argument('--crop-10', dest='crop_10', action='store_true')
    parser.add_argument('--down-ratio', dest='down_ratio', type=int, default=8,
                        help='model downsampling ratio')
    parser.add_argument('--step-ratio', dest='step_ratio', default=0.1,
                        type=float)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--random-color', action='store_true', default=False)
    parser.add_argument('--min-area-ratio', default=0.08, type=float)
    parser.add_argument('--aspect-ratio', type=float, default=4./3)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(' '.join(sys.argv))

    return args


def main():
    args = parse_args()
    print(args)
    if args.cmd == 'train':
        run_training(args)
    elif args.cmd == 'test':
        test_model(args)


def run_training(args):
    model = dla.__dict__[args.arch](
        pretrained=args.pretrained, num_classes=args.classes,
        pool_size=args.crop_size // 32)
    model = torch.nn.DataParallel(model)

    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    data = dataset.get_data(args.data_name)
    if data is None:
        data = dataset.load_dataset_info(args.data, data_name=args.data_name)
    if data is None:
        raise ValueError('{} is not pre-defined in dataset.py and info.json '
                         'does not exist in {}', args.data_name, args.data)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = data_transforms.Normalize(mean=data.mean, std=data.std)
    tt = [data_transforms.RandomResizedCrop(
        args.crop_size, min_area_ratio=args.min_area_ratio,
        aspect_ratio=args.aspect_ratio)]
    if data.eigval is not None and data.eigvec is not None \
            and args.random_color:
        ligiting = data_transforms.Lighting(0.1, data.eigval, data.eigvec)
        jitter = data_transforms.RandomJitter(0.4, 0.4, 0.4)
        tt.extend([jitter, ligiting])
    tt.extend([data_transforms.RandomHorizontalFlip(),
               data_transforms.ToTensor(),
               normalize])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, data_transforms.Compose(tt)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(args.scale_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    if args.evaluate:
        validate(args, val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = 'checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % args.check_freq == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)


def test_model(args):
    # create model
    model = dla.__dict__[args.arch](pretrained=args.pretrained,
                                    pool_size=args.crop_size // 32)
    model = torch.nn.DataParallel(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {} prec {:.03f}) "
                  .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    data = dataset.get_data(args.data_name)
    if data is None:
        data = dataset.load_dataset_info(args.data, data_name=args.data_name)
    if data is None:
        raise ValueError('{} is not pre-defined in dataset.py and info.json '
                         'does not exist in {}', args.data_name, args.data)
    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=data.mean, std=data.std)

    if args.crop_10:
        t = transforms.Compose([
            transforms.Resize(args.scale_size),
            transforms.ToTensor(),
            normalize])
    else:
        t = transforms.Compose([
            transforms.Resize(args.scale_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            normalize])
    val_loader = torch.utils.data.DataLoader(
        ImageFolder(valdir, t, out_name=args.crop_10),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    if args.crop_10:
        validate_10(args, val_loader, model,
                    '{}_i_{}_c_10.txt'.format(args.arch, args.start_epoch))
    else:
        validate(args, val_loader, model, criterion)


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.cuda:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.cuda:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def sample_10(image, crop_dims):
    # Dimensions and center.
    image = image.numpy()
    im_shape = np.array(image.shape[2:])
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
        crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10, image.shape[1], crop_dims[0], crop_dims[1]),
                     dtype=np.float32)
    ix = 0
    for crop in crops_ix:
        crops[ix] = image[0, :, crop[0]:crop[2], crop[1]:crop[3]]
        ix += 1
    crops[ix-5:ix] = crops[ix-5:ix, :, :, ::-1]  # flip for mirrors
    return torch.from_numpy(crops)


def validate_10(args, data_loader, model, out_path):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    sm = nn.functional.softmax
    criterion = nn.NLLLoss()
    out_fp = open(out_path, 'w')
    end = time.time()
    for i, (input, target, name) in enumerate(data_loader):
        assert input.size(0) == 1
        input = sample_10(input, (224, 224))
        if args.cuda:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        output = sm(output)
        output = torch.mean(output, 0)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        _, pred = output.topk(10, 1, True, True)
        pred = pred.view(-1).data.cpu().numpy()
        output = output.view(-1).data.cpu().numpy()
        print(name[0], ','.join("{},{:.03f}".format(pred[i], output[pred[i]])
                                for i in range(10)),
              sep=',', file=out_fp, flush=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(data_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    out_fp.close()


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


def adjust_learning_rate(args, optimizer, epoch):
    if args.lr_adjust == 'step':
        """Sets the learning rate to the initial LR decayed by 10
        every 30 epochs"""
        lr = args.lr * (args.step_ratio ** (epoch // 30))
    else:
        raise ValueError()
    print('Epoch [{}] Learning rate: {:0.6f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

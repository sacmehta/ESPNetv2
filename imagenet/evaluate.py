import argparse
import time
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import Model as Net
import numpy as np
from utils import *
import os
import torchvision.models as preModels

cudnn.benchmark = True

'''
This file is mostly adapted from the PyTorch ImageNet example
'''

def main(args):

    model = Net.EESPNet(classes=1000, s=args.s)
    model = torch.nn.DataParallel(model).cuda()
    if not os.path.isfile(args.weightFile):
        print('Weight file does not exist')
        exit(-1)
    dict_model = torch.load(args.weightFile)
    model.load_state_dict(dict_model)


    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Parameters: ' + str(n_params))

    #if args.parallel:
    #    model = torch.nn.DataParallel(model).cuda()



    #dict_model = torch.load(args.weightFile)
    #model.load_state_dict(dict_model['state_dict'])

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(args.inpSize/0.875)),
            transforms.CenterCrop(args.inpSize),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(int(args.inpSize/0.875)),
            transforms.CenterCrop(args.inpSize),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

#    train_prec, train_top5 = validate(train_loader, model)
    val_prec, val_top5 = validate(val_loader, model)
  #  print('train top1: {}, val top1: {}, train top5: {}, val top5: {}'.format(train_prec, val_prec, train_top5, val_top5))
    return


def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # with torch.no_grad():
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            # target = target.cuda(async=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            ### Uncomment if version is less than 0.4
            #input = Variable(input, volatile=True)
            #target = Variable(target, volatile=True)

            # compute output
            output = model(input)
            loss = loss_fn(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            # replace if using pytorch version < 0.4
            #losses.update(loss.data[0], input.size(0))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print("Batch:[%d/%d]\t\tBatchTime:%.3f\t\tLoss:%.3f\t\ttop1:%.3f (%.3f)\t\ttop5:%.3f(%.3f)" %
                      (i, len(val_loader), batch_time.avg, losses.avg, top1.val, top1.avg, top5.val, top5.avg))

        print(' * Prec@1:%.3f Prec@5:%.3f' % (top1.avg, top5.avg))

        return top1.avg, top5.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESPNetv2 Training on the ImageNet')
    parser.add_argument('--data', default='/home/ubuntu/ILSVRC2015/Data/CLS-LOC/', help='path to dataset')
    parser.add_argument('--workers', default=12, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--s', default=1, type=float,
                        help='Factor by which output channels should be reduced (s > 1 for increasing the dims while < 1 for decreasing)')
    parser.add_argument('--weightFile', type=str, default='', help='weight file')
    parser.add_argument('--inpSize', default=224, type=int,
                        help='Input size')


    args = parser.parse_args()
    args.parallel = True

    main(args)

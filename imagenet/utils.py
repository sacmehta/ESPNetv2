import os
import torch
import shutil
import torch.nn.functional as F
import time


#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

'''
This file is mostly adapted from the PyTorch ImageNet example
'''

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

'''
Utility to save checkpoint or not
'''
def save_checkpoint(state, is_best, back_check, epoch, dir):
    check_pt_file = dir + os.sep + 'checkpoint.pth.tar'
    torch.save(state, check_pt_file)
    if is_best:
        #We only need best models weight and not check point states, etc.
        torch.save(state['state_dict'], dir + os.sep + 'model_best.pth')
    if back_check:
        shutil.copyfile(check_pt_file, dir + os.sep + 'checkpoint_back' + str(epoch) + '.pth.tar')

'''
Cross entropy loss function
'''
def loss_fn(outputs, labels):

    return F.cross_entropy(outputs, labels)

'''
Training loop
'''
def train(train_loader, model, optimizer, epoch):

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

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)

        # compute loss
        loss = loss_fn(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        #losses.update(loss.data[0], input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0: #print after every 100 batches
            print("Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\ttop1:%.4f (%.4f)\t\ttop5:%.4f (%.4f)" %
                  (epoch, i, len(train_loader), batch_time.avg, losses.avg, top1.val, top1.avg, top5.val, top5.avg))


    return top1.avg, losses.avg

'''
Validation loop
'''
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
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

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

            if i % 100 == 0: # print after every 100 batches
                print("Batch:[%d/%d]\t\tBatchTime:%.3f\t\tLoss:%.3f\t\ttop1:%.3f (%.3f)\t\ttop5:%.3f(%.3f)" %
                      (i, len(val_loader), batch_time.avg, losses.avg, top1.val, top1.avg, top5.val, top5.avg))

        print(' * Prec@1:%.3f Prec@5:%.3f' % (top1.avg, top5.avg))

        return top1.avg, losses.avg
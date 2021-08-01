import math
import os
import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import subprocess
from models import GetModel, ESRGAN_Discriminator, ESRGAN_FeatureExtractor
from datahandler import GetDataloaders

from plotting import testAndMakeCombinedPlots, generate_convergence_plots

from torch.utils.tensorboard import SummaryWriter

from options import parser
import traceback
import socket
from datetime import datetime
import shutil
import sys
import glob


def options():

    opt = parser.parse_args()

    if opt.norm == '':
        opt.norm = opt.dataset
    elif opt.norm.lower() == 'none':
        opt.norm = None

    if len(opt.basedir) > 0:
        opt.root = opt.root.replace('basedir', opt.basedir)
        opt.weights = opt.weights.replace('basedir', opt.basedir)
        opt.out = opt.out.replace('basedir', opt.basedir)

    if opt.out[:4] == 'root':
        opt.out = opt.out.replace('root', opt.root)


    # convenience function
    if len(opt.weights) > 0 and not os.path.isfile(opt.weights):
        # folder provided, trying to infer model options

        logfile = opt.weights + '/log.txt'
        opt.weights += '/final.pth'
        if not os.path.isfile(opt.weights):
            opt.weights = opt.weights.replace('final.pth', 'prelim.pth')

        if os.path.isfile(logfile):
            fid = open(logfile, 'r')
            optstr = fid.read()
            optlist = optstr.split(', ')

            def getopt(optname, typestr):
                opt_e = [e.split('=')[-1].strip("\'")
                        for e in optlist if (optname.split('.')[-1] + '=') in e]
                return eval(optname) if len(opt_e) == 0 else typestr(opt_e[0])

            opt.model = getopt('opt.model', str)
            opt.nch_in = getopt('opt.nch_in', int)
            opt.nch_out = getopt('opt.nch_out', int)
            opt.n_resgroups = getopt('opt.n_resgroups', int)
            opt.n_resblocks = getopt('opt.n_resblocks', int)
            opt.n_feats = getopt('opt.n_feats', int)

    return opt


def remove_dataparallel_wrapper(state_dict):
    r"""Converts a DataParallel model to a normal one by removing the "module."
    wrapper in the module dictionary

    Args:
            state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict



def train(opt, dataloader, validloader, net):
    start_epoch = 0
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    loss_function.cuda()
    if len(opt.weights) > 0:  # load previous weights?
        checkpoint = torch.load(opt.weights)
        print('loading checkpoint', opt.weights)

        net.load_state_dict(checkpoint['state_dict'])
        if opt.lr == 1:  # continue as it was
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    if len(opt.scheduler) > 0:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08)
        stepsize, gamma = int(opt.scheduler.split(
            ',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, stepsize, gamma=gamma)
        if len(opt.weights) > 0:
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])

    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, opt.nepoch):
        count = 0
        mean_loss = 0

        # for param_group in optimizer.param_groups:
        #     print('\nLearning rate', param_group['lr'])

        for i, bat in enumerate(dataloader):
            lr, hr = bat[0], bat[1]

            optimizer.zero_grad()

            sr = net(lr.to(opt.device))
            loss = loss_function(sr, hr.to(opt.device))

            loss.backward()
            optimizer.step()

            ######### Status and display #########
            mean_loss += loss.data.item()
            print('\r[%d/%d][%d/%d] Loss: %0.6f' % (epoch+1, opt.nepoch,
                                                    i+1, len(dataloader), loss.data.item()), end='')

            count += 1
            if opt.log and count*opt.batchSize // 1000 > 0:
                t1 = time.perf_counter() - opt.t0
                mem = torch.cuda.memory_allocated()
                opt.writer.add_scalar(
                    'data/mean_loss_per_1000', mean_loss / count, epoch)
                opt.writer.add_scalar('data/time_per_1000', t1, epoch)
                print(epoch, count*opt.batchSize, t1, mem,
                      mean_loss / count, file=opt.train_stats)
                opt.train_stats.flush()
                count = 0

        # ---------------- Scheduler -----------------
        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('\nLearning rate', param_group['lr'])
                break

        # ---------------- Printing -----------------
        mean_loss = mean_loss / len(dataloader)
        t1 = time.perf_counter() - opt.t0
        eta = (opt.nepoch - (epoch + 1)) * t1 / (epoch + 1)
        ostr = '\nEpoch [%d/%d] done, mean loss: %0.6f, time spent: %0.1fs, ETA: %0.1fs' % (
            epoch+1, opt.nepoch, mean_loss, t1, eta)
        print(ostr)
        print(ostr, file=opt.fid)
        opt.fid.flush()
        if opt.log:
            opt.writer.add_scalar(
                'data/mean_loss', mean_loss / len(dataloader), epoch)

        # ---------------- TEST -----------------
        if (epoch + 1) % opt.testinterval == 0:
            testAndMakeCombinedPlots(net, validloader, opt, epoch)


        if (epoch + 1) % opt.saveinterval == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}
            if len(opt.scheduler) > 0:
                checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, '%s/prelim%d.pth' % (opt.out, epoch+1))

    checkpoint = {'epoch': opt.nepoch,
                  'state_dict': net.state_dict(),
                  'optimizer': optimizer.state_dict()}
    if len(opt.scheduler) > 0:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, opt.out + '/final.pth')




def main(opt):
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    os.makedirs(opt.out,exist_ok=True)
    shutil.copy2('options.py',opt.out)

    opt.fid = open(opt.out + '/log.txt', 'w')

    ostr = 'ARGS: ' + ' '.join(sys.argv[:])
    print(opt, '\n')
    print(opt, '\n', file=opt.fid)
    print('\n%s\n' % ostr)
    print('\n%s\n' % ostr, file=opt.fid)


    print('getting dataloader', opt.root)
    dataloader, validloader = GetDataloaders(opt)

    if opt.log:
        opt.writer = SummaryWriter(log_dir=opt.out, comment='_%s_%s' % (
            opt.out.replace('\\', '/').split('/')[-1], opt.model))
        opt.train_stats = open(opt.out.replace(
            '\\', '/') + '/train_stats.csv', 'w')
        opt.test_stats = open(opt.out.replace(
            '\\', '/') + '/test_stats.csv', 'w')
        print('iter,nsample,time,memory,meanloss', file=opt.train_stats)
        print('iter,time,memory,psnr,ssim', file=opt.test_stats)

    t0 = time.perf_counter()
    net = GetModel(opt)

    if not opt.test:
        train(opt, dataloader, validloader, net)
        # torch.save(net.state_dict(), opt.out + '/final.pth')
    else:
        if len(opt.weights) > 0:  # load previous weights?
            checkpoint = torch.load(opt.weights)
            print('loading checkpoint', opt.weights)
            net.load_state_dict(checkpoint['state_dict'])
            print('time: %0.1f' % (time.perf_counter()-t0))
        testAndMakeCombinedPlots(net, validloader, opt)

    opt.fid.close()
    if not opt.test:
        generate_convergence_plots(opt,opt.out + '/log.txt')


    print('time: %0.1f' % (time.perf_counter()-t0))

    # optional clean up
    if opt.disposableTrainingData and not opt.test:
        print('deleting training data')
        # preserve a few samples
        os.makedirs('%s/training_data_subset' % opt.out, exist_ok=True)

        samplecount = 0
        for file in glob.glob('%s/*' % opt.root):
            if os.path.isfile(file):
                basename = os.path.basename(file)
                shutil.copy2(file, '%s/training_data_subset/%s' % (opt.out,basename))
                samplecount += 1
                if samplecount == 10:
                    break
        shutil.rmtree(opt.root)


if __name__ == '__main__':
    opt = options()
    main(opt)

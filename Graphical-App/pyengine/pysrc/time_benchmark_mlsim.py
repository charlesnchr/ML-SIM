from mlsim_models import GetModel
from PIL import Image
import numpy as np
import datetime
import math
import os

import torch
import time

import skimage.io
import skimage.transform
import cv2
import glob

import torch.optim as optim
from skimage import exposure
from argparse import Namespace
import tifffile as tiff
import torch.nn.functional as F



# toTensor = F.to_tensor()
# toPIL = F.to_pil_image()


def GetOptions():
    # training options
    opt = Namespace()
    opt.model = 'rcan'
    opt.n_resgroups = 3
    opt.n_resblocks = 10
    opt.n_feats = 96
    opt.reduction = 16
    opt.narch = 0
    opt.norm = 'minmax'

    opt.cpu = False
    opt.multigpu = False
    opt.undomulti = False

    opt.imageSize = 512
    opt.weights = "model/simrec_simin_gtout_rcan_512_2_ntrain790-final.pth"

    opt.task = 'simin_gtout'
    opt.scale = 1
    opt.nch_in = 9
    opt.nch_out = 1

    return opt


def GetOptions_allRnd():
    # training options
    opt = Namespace()
    opt.model = 'rcan'
    opt.n_resgroups = 3
    opt.n_resblocks = 10
    opt.n_feats = 48
    opt.reduction = 16
    opt.narch = 0
    opt.norm = 'adapthist'
    opt.cmap = 'viridis'

    opt.cpu = False
    opt.multigpu = False
    opt.undomulti = False

    opt.imageSize = 512
    opt.weights = "../models/0216_SIMRec_0214_rndAll_rcan_continued.pth"

    opt.task = 'simin_gtout'
    opt.scale = 1
    opt.nch_in = 9
    opt.nch_out = 1

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


def LoadModel(opt):
    print('Loading model')
    print(opt)

    net = GetModel(opt)
    print('loading checkpoint', opt.weights)
    checkpoint = torch.load(opt.weights)

    if type(checkpoint) is dict:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if opt.undomulti:
        state_dict = remove_dataparallel_wrapper(state_dict)
    net.load_state_dict(state_dict)

    return net


def prepimg(stack, self):

    inputimg = stack[:9]

    if self.nch_in == 6:
        inputimg = inputimg[[0, 1, 3, 4, 6, 7]]
    elif self.nch_in == 3:
        inputimg = inputimg[[0, 4, 8]]

    if inputimg.shape[1] > 512 or inputimg.shape[2] > 512:
        print('Over 512x512! Cropping')
        inputimg = inputimg[:, :512, :512]
    if self.norm == 'convert':  # raw img from microscope, needs normalisation and correct frame ordering
        print('Raw input assumed - converting')
        # NCHW
        # I = np.zeros((9,opt.imageSize,opt.imageSize),dtype='uint16')

        # for t in range(9):
        #     frame = inputimg[t]
        #     frame = 120 / np.max(frame) * frame
        #     frame = np.rot90(np.rot90(np.rot90(frame)))
        #     I[t,:,:] = frame
        # inputimg = I

        inputimg = np.rot90(inputimg, axes=(1, 2))
        # could also do [8,7,6,5,4,3,2,1,0]
        inputimg = inputimg[[6, 7, 8, 3, 4, 5, 0, 1, 2]]
        for i in range(len(inputimg)):
            inputimg[i] = 100 / np.max(inputimg[i]) * inputimg[i]
    elif 'convert' in self.norm:
        fac = float(self.norm[7:])
        inputimg = np.rot90(inputimg, axes=(1, 2))
        # could also do [8,7,6,5,4,3,2,1,0]
        inputimg = inputimg[[6, 7, 8, 3, 4, 5, 0, 1, 2]]
        for i in range(len(inputimg)):
            inputimg[i] = fac * 255 / np.max(inputimg[i]) * inputimg[i]
    inputimg = inputimg.astype('float') / np.max(inputimg)  # used to be /255
    widefield = np.mean(inputimg, 0)

    if self.norm == 'adapthist':
        for i in range(len(inputimg)):
            inputimg[i] = exposure.equalize_adapthist(
                inputimg[i], clip_limit=0.001)
        widefield = exposure.equalize_adapthist(widefield, clip_limit=0.001)
    else:
        # normalise
        inputimg = torch.tensor(inputimg).float()
        widefield = torch.tensor(widefield).float()
        widefield = (widefield - torch.min(widefield)) / \
            (torch.max(widefield) - torch.min(widefield))

        if self.norm == 'minmax':
            for i in range(len(inputimg)):
                inputimg[i] = (inputimg[i] - torch.min(inputimg[i])) / \
                    (torch.max(inputimg[i]) - torch.min(inputimg[i]))
        elif 'minmax' in self.norm:
            fac = float(self.norm[6:])
            for i in range(len(inputimg)):
                inputimg[i] = fac * (inputimg[i] - torch.min(inputimg[i])) / \
                    (torch.max(inputimg[i]) - torch.min(inputimg[i]))

    # otf = torch.tensor(otf.astype('float') / np.max(otf)).unsqueeze(0).float()
    # gt = torch.tensor(gt.astype('float') / 255).unsqueeze(0).float()
    # simimg = torch.tensor(simimg.astype('float') / 255).unsqueeze(0).float()
    # widefield = torch.mean(inputimg,0).unsqueeze(0)

    # normalise
    # gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
    # simimg = (simimg - torch.min(simimg)) / (torch.max(simimg) - torch.min(simimg))
    # widefield = (widefield - torch.min(widefield)) / (torch.max(widefield) - torch.min(widefield))
    inputimg = torch.tensor(inputimg).float()
    widefield = torch.tensor(widefield).float()
    return inputimg, widefield



def EvaluateModel(net, opt, stackdir, outfile):

    inputtensors = None
    count = 0

    for filepath in glob.glob('%s/*.tif' % stackdir):
        stack = tiff.imread(filepath, key=range(9))
        inputimg, widefield = prepimg(stack, opt)
        inputimg = inputimg.cuda()
        print('loaded %s' % filepath)

        if inputtensors is None:
            inputtensors = inputimg.unsqueeze(0)
        else:
            inputtensors = torch.cat([inputtensors, inputimg.unsqueeze(0)], dim=0)

        if count == 24:
            break
        count += 1

    t0 = time.perf_counter()

    print('tensors', inputtensors.shape)

    with torch.no_grad():
        sr = net(inputtensors)
        sr = sr.cpu()
        sr = torch.clamp(sr, min=0, max=1)

        # pil_img = toPIL(sr[0])
        # print(np_img.shape)
        # pil_img = Image.fromarray((np_img*255).astype('uint8'))

    print('time final', time.perf_counter()-t0)

    for fidx in range(sr.shape[0]):
        oimg = sr[fidx].squeeze()
        oimg = oimg.numpy()
        oimg = (255*oimg).astype('uint8')
        cv2.imwrite('%s_%d.png' % (outfile,fidx), oimg)

    # should ideally be done by drawing on client side, in javascript
    # save_image(sr_img, '%s_sr.png' % outfile, cmap)



def EvaluateModel_rep(net, opt, stackdir, outfile):

    inputtensors = None
    count = 0
    N = 20

    for filepath in glob.glob('%s/*.tif' % stackdir):
        stack = tiff.imread(filepath, key=range(9))
        inputimg, widefield = prepimg(stack, opt)
        inputimg = inputimg.cuda()
        
        print('loaded %s' % filepath)

        inputtensors = inputimg.unsqueeze(0)

        for i in range(N):
            inputtensors = torch.cat([inputtensors, inputimg.unsqueeze(0)], dim=0)
    
        break


    

    print('tensors', inputtensors.shape)

    # warm up
    for i in range(3): 
        with torch.no_grad():
            sr = net(inputtensors)


    # actual timing

    tarr = []
    
    for i in range(24): # 24 to get a similar sample size as the other timings
        t0 = time.perf_counter()
            
        with torch.no_grad():
            sr = net(inputtensors)


            # pil_img = toPIL(sr[0])
            # print(np_img.shape)
            # pil_img = Image.fromarray((np_img*255).astype('uint8'))

        tdelta = time.perf_counter()-t0
        tarr.append(tdelta/N)
        print('time final per image', tdelta/N)


    sr = torch.clamp(sr, min=0, max=1)
    sr = sr.cpu()

    # save only once
    for fidx in range(sr.shape[0]):
        oimg = sr[fidx].squeeze()
        oimg = oimg.numpy()
        oimg = (255*oimg).astype('uint8')
        cv2.imwrite('%s_%d.png' % (outfile,fidx), oimg)


    print('STATISTICS')
    tarr = np.array(tarr)
    print(tarr)
    print(np.mean(tarr))
    print(np.std(tarr))

    # should ideally be done by drawing on client side, in javascript
    # save_image(sr_img, '%s_sr.png' % outfile, cmap)
    

def reconstruct(exportdir, dirpath):
    opt = GetOptions_allRnd()
    model = LoadModel(opt)
    print('loaded model')

    os.makedirs(exportdir, exist_ok=True)

    outfile = '%s/%s' % (exportdir,
                         datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3])

    # EvaluateModel(model, opt, dirpath, outfile)
    
    EvaluateModel_rep(model, opt, dirpath, outfile)
    

    



reconstruct('C:/Users/charl/Desktop/MLSIM-kodak-out',
            'C:/Users/charl/Desktop/ML-SIM-kodak')

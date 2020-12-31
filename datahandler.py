import os

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
from PIL import Image, ImageOps
import random

import numpy as np

from skimage import io, exposure, transform


def PSNR(I0,I1):
    MSE = torch.mean( (I0-I1)**2 )
    PSNR = 20*torch.log10(1/torch.sqrt(MSE))
    return PSNR

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(48),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

normalize2 = transforms.Normalize(mean = [0.69747254,0.53480325,0.68800158], std = [0.23605522,0.27857294,0.21456957])
unnormalize2 = transforms.Normalize(mean = [-2.9547, -1.9198, -3.20643], std = [4.2363, 3.58972, 4.66049])


toTensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage()      


def GetDataloaders(opt):

    # dataloaders
    if opt.dataset.lower() == 'fouriersim': 
        dataloader = load_fourier_SIM_dataset(opt.root,'train',opt)
        validloader = load_fourier_SIM_dataset(opt.root,'valid',opt)
    else:
        print('unknown dataset')
        return None,None
    return dataloader, validloader



class Fourier_SIM_dataset(Dataset):

    def __init__(self, root, category, opt):

        self.images = []
        for folder in root.split(','):
            folderimgs = glob.glob(folder + '/*.tif')
            self.images.extend(folderimgs)

        random.seed(1234)
        random.shuffle(self.images)

        if category == 'train':
            self.images = self.images[:opt.ntrain]
        else:
            self.images = self.images[-opt.ntest:]

        self.len = len(self.images)
        self.scale = opt.scale
        self.task = opt.task
        self.nch_in = opt.nch_in
        self.nch_out = opt.nch_out
        self.norm = opt.norm
        self.out = opt.out

    def __getitem__(self, index):
        
        stack = io.imread(self.images[index])

        if self.nch_in == 6:
            inputimg = stack[[0,1,3,4,6,7]]
        elif self.nch_in == 3:
            inputimg = stack[[0,4,8]]
        elif self.nch_in == 1:
            inputimg = stack[[8]] # used for sequential SIM - first tests from 20201215 have GT as 9th frame
        else:
            inputimg = stack[:self.nch_in]


        # adding noise
        # if 'noiseRetraining' in self.out:
        #     noisefrac = np.linspace(0,1,10)
        #     idx = np.random.randint(0,10)
        #     inputimg = inputimg + noisefrac[idx]*np.std(I)*np.random.randn(*inputimg.shape)
        #     inputimg = np.clip(inputimg,0,255).astype('uint16')


        if len(stack) > 9:
            # otf = stack[9]
            if self.scale == 2:
                toprow = np.hstack((stack[-4,:,:],stack[-2,:,:]))
                botrow = np.hstack((stack[-3,:,:],stack[-1,:,:]))
                gt = np.vstack((toprow,botrow)).reshape(2*stack.shape[1],2*stack.shape[2])
            elif self.nch_out > 1:
                gt = stack[-self.nch_out:]
            else:
                gt = stack[-1] # used to be index self.nch_in+1
        else:
            gt = stack[0] # if it doesn't exist, doesn't matter


        # widefield = stack[12]

        # print('max before:',end=' ')
        # print('%0.2f %0.2f %0.2f %0.2f %0.2f' % (np.max(inputimg),np.max(otf),np.max(gt),np.max(simimg),np.max(widefield)))

        if self.norm == 'convert': # raw img from microscope, needs normalisation and correct frame ordering
            print('Raw input assumed - converting')
            # NCHW
            # I = np.zeros((9,opt.imageSize,opt.imageSize),dtype='uint16')

            # for t in range(9):
            #     frame = inputimg[t]
            #     frame = 120 / np.max(frame) * frame
            #     frame = np.rot90(np.rot90(np.rot90(frame)))
            #     I[t,:,:] = frame
            # inputimg = I

            inputimg = np.rot90(inputimg,axes=(1,2))
            inputimg = inputimg[[6,7,8,3,4,5,0,1,2]] # could also do [8,7,6,5,4,3,2,1,0]
            for i in range(len(inputimg)):
                inputimg[i] = 100 / np.max(inputimg[i]) * inputimg[i]
        elif 'convert' in self.norm:
            fac = float(self.norm[7:])
            inputimg = np.rot90(inputimg,axes=(1,2))
            inputimg = inputimg[[6,7,8,3,4,5,0,1,2]] # could also do [8,7,6,5,4,3,2,1,0]
            for i in range(len(inputimg)):
                inputimg[i] = fac * 255 / np.max(inputimg[i]) * inputimg[i]


        inputimg = inputimg.astype('float') / np.max(inputimg) # used to be /255
        gt = gt.astype('float') / np.max(gt) # used to be /255
        widefield = np.mean(inputimg,0)

        if len(stack) > self.nch_in+2:
            simimg = stack[self.nch_in+2] # sim reference image
            simimg = simimg.astype('float') / np.max(simimg)
        else:
            simimg = np.mean(inputimg,0) # same as widefield
        
        if self.norm == 'adapthist':
            for i in range(len(inputimg)):
                inputimg[i] = exposure.equalize_adapthist(inputimg[i],clip_limit=0.001)
            widefield = exposure.equalize_adapthist(widefield,clip_limit=0.001)
            gt = exposure.equalize_adapthist(gt,clip_limit=0.001)
            simimg = exposure.equalize_adapthist(simimg,clip_limit=0.001)

            inputimg = torch.tensor(inputimg).float()
            gt = torch.tensor(gt).unsqueeze(0).float()
            widefield = torch.tensor(widefield).unsqueeze(0).float()
            simimg = torch.tensor(simimg).unsqueeze(0).float()
        else:
            inputimg = torch.tensor(inputimg).float()
            gt = torch.tensor(gt).float()
            if self.nch_out == 1:
                gt = gt.unsqueeze(0)
            widefield = torch.tensor(widefield).unsqueeze(0).float()
            simimg = torch.tensor(simimg).unsqueeze(0).float()

            # normalise 
            gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
            simimg = (simimg - torch.min(simimg)) / (torch.max(simimg) - torch.min(simimg))
            widefield = (widefield - torch.min(widefield)) / (torch.max(widefield) - torch.min(widefield))

            if self.norm == 'minmax':
                for i in range(len(inputimg)):
                    inputimg[i] = (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))
            elif 'minmax' in self.norm:
                fac = float(self.norm[6:])
                for i in range(len(inputimg)):
                    inputimg[i] = fac * (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))
        

        if self.task == 'simin_simout':
            return inputimg,simimg,gt,widefield   # sim input, sim output
        elif self.task == 'wfin_simout':
            return widefield,simimg,gt,widefield   # wf input, sim output
        elif self.task == 'wfin_gtout':
            return widefield,gt,simimg,widefield  # wf input, gt output
        else: # simin_gtout
            return inputimg,gt,simimg,widefield  # sim input, gt output



    def __len__(self):
        return self.len        

def load_fourier_SIM_dataset(root, category,opt):

    dataset = Fourier_SIM_dataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader    



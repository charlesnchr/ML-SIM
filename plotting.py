import torch
import matplotlib.pyplot as plt
import torchvision
import skimage
from skimage.metrics import structural_similarity
# from skimage.measure import compare_ssim
import torchvision.transforms as transforms
import numpy as np
import time
from PIL import Image
import scipy.ndimage as ndimage
import torch.nn as nn
import os


plt.switch_backend('agg')

toTensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage()      

def testAndMakeCombinedPlots(net,loader,opt,idx=0):

    def PSNR_numpy(p0,p1):
        I0,I1 = np.array(p0)/255.0, np.array(p1)/255.0
        MSE = np.mean( (I0-I1)**2 )
        PSNR = 20*np.log10(1/np.sqrt(MSE))
        return PSNR

    def SSIM_numpy(p0,p1):
        I0,I1 = np.array(p0)/255.0, np.array(p1)/255.0
        return structural_similarity(I0, I1, multichannel=True)
        # return compare_ssim(I0, I1, multichannel=True)

    def calcScores(img, hr=None, makeplotBool=False, plotidx=0, title=None):
        if makeplotBool:
            plt.subplot(1,3,plotidx)
            plt.gca().axis('off')
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(img,cmap='gray')
        if not hr == None:
            psnr,ssim = PSNR_numpy(img,hr),SSIM_numpy(img,hr)
            if makeplotBool: plt.title('%s (%0.2fdB/%0.3f)' % (title,psnr,ssim))
            return psnr,ssim
        if makeplotBool: plt.title(r'GT ($\infty$/1.000)')


    count, mean_bc_psnr, mean_sr_psnr, mean_bc_ssim, mean_sr_ssim = 0,0,0,0,0

    for i, bat in enumerate(loader):
        lr_bat, hr_bat = bat[0], bat[1]
        with torch.no_grad():
            sr_bat = net(lr_bat.to(opt.device))
        sr_bat = sr_bat.cpu()

        for j in range(len(lr_bat)): # loop over batch
            makeplotBool = (idx < 5 or (idx+1) % opt.plotinterval == 0 or idx == opt.nepoch - 1) and count < opt.nplot
            if opt.test: makeplotBool = True

            lr, sr, hr = lr_bat.data[j], sr_bat.data[j], hr_bat.data[j]

            if opt.task == 'simin_simout' or opt.task == 'wfin_simout':
                ## sim target
                gt_bat = bat[2]
                wf_bat = bat[3]
                bc, hr, lr = hr_bat.data[j], gt_bat.data[j], wf_bat.data[j]
                sr = torch.clamp(sr,min=0,max=1)     
            else: 
                ## gt target
                sim_bat = bat[2]
                wf_bat = bat[3]
                bc, hr, lr = sim_bat.data[j], hr_bat.data[j], wf_bat.data[j]
                sr = torch.clamp(sr,min=0,max=1) 

            # fix to deal with 3D deconvolution
            if opt.nch_out > 1:
                lr = lr[lr.shape[0] // 2] # channels are not for colours but separate grayscale frames, take middle
                sr = sr[sr.shape[0] // 2]
                hr = hr[hr.shape[0] // 2]

            ### Common commands
            lr, bc, sr, hr = toPIL(lr), toPIL(bc), toPIL(sr), toPIL(hr)

            if opt.scale == 2:
                lr = lr.resize((1024,1024), resample=Image.BICUBIC)
                bc = bc.resize((1024,1024), resample=Image.BICUBIC)
                hr = hr.resize((1024,1024), resample=Image.BICUBIC)

            if makeplotBool: plt.figure(figsize=(10,5),facecolor='white')
            bc_psnr, bc_ssim = calcScores(lr, hr, makeplotBool, plotidx=1, title='WF')
            sr_psnr, sr_ssim = calcScores(sr, hr, makeplotBool, plotidx=2, title='SR')
            calcScores(hr, None, makeplotBool, plotidx=3)
            
            mean_bc_psnr += bc_psnr
            mean_sr_psnr += sr_psnr
            mean_bc_ssim += bc_ssim
            mean_sr_ssim += sr_ssim

            if makeplotBool:
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                plt.savefig('%s/combined_epoch%d_%d.png' % (opt.out,idx+1,count), dpi=300, bbox_inches = 'tight', pad_inches = 0)
                plt.close()
                if opt.test:
                    lr.save('%s/lr_epoch%d_%d.png' % (opt.out,idx+1,count))
                    sr.save('%s/sr_epoch%d_%d.png' % (opt.out,idx+1,count))
                    hr.save('%s/hr_epoch%d_%d.png' % (opt.out,idx+1,count))

            count += 1
            if count == opt.ntest: break
        if count == opt.ntest: break
    
    summarystr = ""
    if count == 0: 
        summarystr += 'Warning: all test samples skipped - count forced to 1 -- '
        count = 1
    summarystr += 'Testing of %d samples complete. bc: %0.2f dB / %0.4f, sr: %0.2f dB / %0.4f' % (count, mean_bc_psnr / count, mean_bc_ssim / count, mean_sr_psnr / count, mean_sr_ssim / count)
    print(summarystr)
    print(summarystr,file=opt.fid)
    opt.fid.flush()
    if opt.log and not opt.test:
        t1 = time.perf_counter() - opt.t0
        mem = torch.cuda.memory_allocated()
        print(idx,t1,mem,mean_sr_psnr / count, mean_sr_ssim / count, file=opt.test_stats)
        opt.test_stats.flush()


def generate_convergence_plots(opt,filename):
    fid = open(filename,'r')
    psnrlist = []
    ssimlist = []

    for line in fid:
        if 'sr: ' in line:
            psnrlist.append(float(line.split('sr: ')[1].split(' dB')[0]))
            ssimlist.append(float(line.split('sr: ')[1].split(' dB / ')[1]))
    
    plt.figure(figsize=(12,5),facecolor='white')
    plt.subplot(121)
    plt.plot(psnrlist,'.-')
    plt.title('PSNR')

    plt.subplot(122)
    plt.plot(ssimlist,'.-')
    plt.title('SSIM')

    plt.savefig('%s/convergencePlot.png' % opt.out, dpi=300)
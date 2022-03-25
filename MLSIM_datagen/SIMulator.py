import sys
import numpy as np
from numpy import pi, cos, sin

from skimage import io, transform
from SIMulator_functions import *
import glob
import os
import argparse
from multiprocessing import Pool

# ------------ Options --------------
nrep = 1
outdir = "PATH_TO_DATA/SIMdata-20201102"
os.makedirs(outdir, exist_ok=True)

# for DIV2k
files = glob.glob("PATH_TO_DATA/DIV2K/DIV2K_train_HR/*.png")
# files = glob.glob("D:/DIV2K/DIV2K_train_HR/*.png")

# single test image
# files = glob.glob('TestImage.png')


# ------------ Parameters-------------
def GetParams(): # uniform randomisation
    opt = argparse.Namespace()

    # phase shifts for each stripe
    opt.Nshifts = 5
    # number of orientations of stripes
    opt.Nangles = 5
    # used to adjust PSF/OTF width
    opt.scale = 0.9 + 0.1*(np.random.rand()-0.5)
    # modulation factor
    opt.ModFac = 0.8 + 0.3*(np.random.rand()-0.5)
    # orientation offset
    opt.alpha = pi/3*(np.random.rand()-0.5)
    # orientation error
    opt.angleError = 10*pi/180*(np.random.rand()-0.5)
    # shuffle the order of orientations
    opt.shuffleOrientations = True
    # random phase shift errors
    opt.phaseError = 1*pi*(0.5-np.random.rand(opt.Nangles, opt.Nshifts))
    # mean illumination intensity
    opt.meanInten = np.ones(opt.Nangles)*0.5
    # amplitude of illumination intensity above mean
    opt.ampInten = np.ones(opt.Nangles)*0.5*opt.ModFac
    # illumination freq
    opt.k2 = 126 + 30*(np.random.rand()-0.5)
    # noise type
    opt.usePoissonNoise = False
    # noise level (percentage for Gaussian)
    opt.NoiseLevel = 8 + 0*8*(np.random.rand()-0.5)
    # 1(to blur using PSF), 0(to blur using OTF)
    opt.UsePSF = 0
    # include OTF and GT in stack
    opt.OTF_and_GT = True

    return opt



# ------------ Main loop --------------
def processImage(file):
    Io = io.imread(file) / 255
    Io = transform.resize(Io, (512, 512), anti_aliasing=True)

    if len(Io.shape) > 2 and Io.shape[2] > 1:
        Io = Io.mean(2)  # if not grayscale

    filename = os.path.basename(file).replace('.png', '')

    print('Generating SIM frames for', file)

    for n in range(nrep):
        opt = GetParams()
        opt.outputname = '%s/%s_%d.tif' % (outdir, filename, n)
        I = Generate_SIM_Image(opt, Io)



if __name__ == '__main__':
    if len(sys.argv) == 1:
        with Pool(40) as p:
            p.map(processImage,files)
    elif sys.argv[1] == 'getstats':
        sumval = 0
        maxval = 0
        minval = 1e10
        for i in range(1000):
            opt = GetParams()
            val = eval('opt.'+sys.argv[2])
            sumval += val
            if val < minval: minval = val
            if val > maxval: maxval = val
        meanval = sumval / 1000
        print('mean of',sys.argv[2],'found to be',meanval)
        print('max of',sys.argv[2],'found to be',maxval)
        print('min of',sys.argv[2],'found to be',minval)

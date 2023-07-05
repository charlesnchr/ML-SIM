import numpy as np
from numpy import pi
from skimage import io
import glob
import os
import argparse
from multiprocessing import Pool, set_start_method

from SIMulator_functions import (
    Generate_SIM_Image,
    cos_wave,
    cos_wave_envelope,
    square_wave_one_third,
    square_wave,
    square_wave_large_spacing
)

np.random.seed(20221219)

def GetParams_20230703(opt):  # uniform randomisation
    SIMopt = argparse.Namespace()

    # modulation factor
    SIMopt.ModFac = opt.ModFac + 0.3*(np.random.rand()-0.5)

    # ---- stripes
    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # orientation offset
    SIMopt.alpha = opt.alphaErrorFac * pi * (np.random.rand() - 0.5)
    # orientation error
    SIMopt.angleError = opt.angleError * pi / 180 * (np.random.rand() - 0.5)
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = not opt.dontShuffleOrientations
    # random phase shift errors
    SIMopt.phaseError = (
        opt.phaseErrorFac * pi * (0.5 - np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
    )
    # illumination freq
    SIMopt.k2 = opt.k2 + opt.k2_err * (np.random.rand() - 0.5)

    # --- spots
    SIMopt.Nspots = opt.Nspots
    SIMopt.spotSize = opt.spotSize

    # used to adjust PSF/OTF width
    SIMopt.PSFOTFscale = opt.PSFOTFscale + 0.2*(np.random.rand()-0.5)
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel + opt.NoiseLevelRandFac * (
        np.random.rand() - 0.5
    )
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination
    SIMopt.noStripes = opt.noStripes

    # function to use for stripes
    SIMopt.func = cos_wave

    SIMopt.patterns = opt.patterns
    SIMopt.crop_factor = opt.crop_factor
    SIMopt.SIMmodality = opt.SIMmodality
    SIMopt.dmdMapping = opt.dmdMapping

    # --- Nframes
    if SIMopt.SIMmodality == "stripes":
        SIMopt.Nframes = SIMopt.Nangles * SIMopt.Nshifts
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = np.ones(SIMopt.Nangles) * SIMopt.ModFac
        # mean illumination intensity
        SIMopt.meanInten = 1 - SIMopt.ampInten
    else:
        SIMopt.Nframes = (SIMopt.Nspots // SIMopt.spotSize) ** 2
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = SIMopt.ModFac
        # mean illumination intensity
        SIMopt.meanInten = 1 - SIMopt.ampInten
        # resize amount of spots (to imitate effect of cropping from FOV on DMD/camera sensor)
        SIMopt.spotResize = 0.7 + 0.6 * (np.random.rand() - 0.5)

    SIMopt.imageSize = opt.imageSize

    return SIMopt




def GetParams_20230625(opt):  # uniform randomisation
    SIMopt = argparse.Namespace()

    # modulation factor
    SIMopt.ModFac = opt.ModFac + 0.3*(np.random.rand()-0.5)

    # ---- stripes
    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # orientation offset
    SIMopt.alpha = opt.alphaErrorFac * pi * (np.random.rand() - 0.5)
    # orientation error
    SIMopt.angleError = opt.angleError * pi / 180 * (np.random.rand() - 0.5)
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = not opt.dontShuffleOrientations
    # random phase shift errors
    SIMopt.phaseError = (
        opt.phaseErrorFac * pi * (0.5 - np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
    )
    # illumination freq
    SIMopt.k2 = opt.k2 + opt.k2_err * (np.random.rand() - 0.5)

    # --- spots
    SIMopt.Nspots = opt.Nspots
    SIMopt.spotSize = opt.spotSize

    # used to adjust PSF/OTF width
    SIMopt.PSFOTFscale = opt.PSFOTFscale + 0.2*(np.random.rand()-0.5)
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel + opt.NoiseLevelRandFac * (
        np.random.rand() - 0.5
    )
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination
    SIMopt.noStripes = opt.noStripes

    # function to use for stripes
    SIMopt.func = np.cos

    SIMopt.patterns = opt.patterns
    SIMopt.crop_factor = opt.crop_factor
    SIMopt.SIMmodality = opt.SIMmodality
    SIMopt.dmdMapping = opt.dmdMapping

    # --- Nframes
    if SIMopt.SIMmodality == "stripes":
        SIMopt.Nframes = SIMopt.Nangles * SIMopt.Nshifts
        # mean illumination intensity
        SIMopt.meanInten = np.ones(SIMopt.Nangles)
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = np.ones(SIMopt.Nangles) * SIMopt.ModFac
    else:
        SIMopt.Nframes = (SIMopt.Nspots // SIMopt.spotSize) ** 2
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = SIMopt.ModFac
        # mean illumination intensity
        SIMopt.meanInten = 1 - SIMopt.ampInten
        # resize amount of spots (to imitate effect of cropping from FOV on DMD/camera sensor)
        SIMopt.spotResize = 0.7 + 0.6 * (np.random.rand() - 0.5)

    SIMopt.imageSize = opt.imageSize

    return SIMopt


def GetParams_20230410(opt):  # uniform randomisation
    SIMopt = argparse.Namespace()

    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # used to adjust PSF/OTF width
    SIMopt.PSFOTFscale = opt.PSFOTFscale + 0.1 * (np.random.rand() - 0.5)
    # modulation factor
    SIMopt.ModFac = opt.ModFac + 0.3 * (np.random.rand() - 0.5)
    # orientation offset
    SIMopt.alpha = opt.alphaErrorFac * pi * (np.random.rand() - 0.5)
    # orientation error
    SIMopt.angleError = opt.angleError * pi / 180 * (np.random.rand() - 0.5)
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = not opt.dontShuffleOrientations
    # random phase shift errors
    SIMopt.phaseError = (
        opt.phaseErrorFac * pi * (0.5 - np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
    )
    # mean illumination intensity
    SIMopt.meanInten = np.ones(SIMopt.Nangles) * 0.5
    # amplitude of illumination intensity above mean
    SIMopt.ampInten = np.ones(SIMopt.Nangles) * 0.5 * SIMopt.ModFac
    # illumination freq
    SIMopt.k2 = opt.k2 + opt.k2_err * (np.random.rand() - 0.5)
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel + opt.NoiseLevelRandFac * (
        np.random.rand() - 0.5
    )
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination
    SIMopt.noStripes = opt.noStripes

    # function to use for stripes
    SIMopt.func = (
        np.cos
        if np.random.rand() < 0.5
        else square_wave_one_third
    )

    return SIMopt


# ------------ Parameters-------------
def GetParams(opt):  # uniform randomisation
    SIMopt = argparse.Namespace()

    # modulation factor
    SIMopt.ModFac = opt.ModFac  # + 0.3*(np.random.rand()-0.5)

    # ---- stripes
    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # orientation offset
    SIMopt.alpha = opt.alphaErrorFac * pi * (np.random.rand() - 0.5)
    # orientation error
    SIMopt.angleError = opt.angleError * pi / 180 * (np.random.rand() - 0.5)
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = not opt.dontShuffleOrientations
    # random phase shift errors
    SIMopt.phaseError = (
        opt.phaseErrorFac * pi * (0.5 - np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
    )
    # illumination freq
    SIMopt.k2 = opt.k2 + opt.k2_err * (np.random.rand() - 0.5)

    # --- spots
    SIMopt.Nspots = opt.Nspots
    SIMopt.spotSize = opt.spotSize

    # used to adjust PSF/OTF width
    SIMopt.PSFOTFscale = opt.PSFOTFscale  # + 0.1*(np.random.rand()-0.5)
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel + opt.NoiseLevelRandFac * (
        np.random.rand() - 0.5
    )
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination
    SIMopt.noStripes = opt.noStripes

    # function to use for stripes
    SIMopt.func = np.cos

    SIMopt.patterns = opt.patterns
    SIMopt.crop_factor = opt.crop_factor
    SIMopt.SIMmodality = opt.SIMmodality
    SIMopt.dmdMapping = opt.dmdMapping

    # --- Nframes
    if SIMopt.SIMmodality == "stripes":
        SIMopt.Nframes = SIMopt.Nangles * SIMopt.Nshifts
        # mean illumination intensity
        SIMopt.meanInten = np.ones(SIMopt.Nangles)
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = np.ones(SIMopt.Nangles) * SIMopt.ModFac
    else:
        SIMopt.Nframes = (SIMopt.Nspots // SIMopt.spotSize) ** 2
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = SIMopt.ModFac
        # mean illumination intensity
        SIMopt.meanInten = 1 - SIMopt.ampInten
        # resize amount of spots (to imitate effect of cropping from FOV on DMD/camera sensor)
        SIMopt.spotResize = 1

    SIMopt.imageSize = opt.imageSize

    return SIMopt


# ------------ Main loop --------------
def processImage(file, opt, SIMopt_override=None):
    if "npy" in opt.ext:
        Io = np.load(file, allow_pickle=True) / 255
        filename = os.path.basename(file).replace(".npy", "")

        if len(Io.shape) > 2 and Io.shape[2] > 3:
            Io = Io[:, :, 8]  # assuming t-stack
        elif Io.shape[2] > 1:
            Io = Io.mean(2)  # if not grayscale
    else:
        Io = io.imread(file) / 255
        # Io = transform.resize(Io, (opt.imageSize, opt.imageSize), anti_aliasing=True)

        if len(Io.shape) > 2 and Io.shape[2] > 1:
            Io = Io.mean(2)  # if not grayscale

        filename = os.path.basename(file).replace(".png", "")

    print("Generating SIM frames for", file)

    gt_dim = opt.imageSize
    if type(gt_dim) is int:
        gt_dim = (gt_dim, gt_dim)

    # multiple by opt.scale
    gt_dim = [int(x * opt.scale) for x in gt_dim]

    for n in range(opt.nrep):
        if SIMopt_override is None:
            SIMopt = eval("%s(opt)" % opt.params)  # GetParams
        else:
            SIMopt = SIMopt_override

        SIMopt.outputname = "%s/%s_%d.tif" % (opt.root, filename, n)

        I = Generate_SIM_Image(
            SIMopt, Io, opt.imageSize, gt_dim, func=SIMopt.func
        )

    return I


class Paralleliser():
    def __init__(self, opt):
        self.opt = opt
        if not opt.seqSIM:
            self.process_func = processImage
        elif "imagefolder" not in opt.ext:
            self.process_func = processSeqImage
        else:
            # processSeqImage if using tif files instead of folders of jpgs
            self.process_func = processSeqImageFolder

    def process(self, file):
        return self.process_func(file, self.opt)


    def run(self, files):
        set_start_method('spawn')
        if self.opt.datagen_workers > 1:
            pool = Pool(processes=self.opt.datagen_workers)
            pool.map(self.process, files)
        else:
            for file in files:
                self.process(file)


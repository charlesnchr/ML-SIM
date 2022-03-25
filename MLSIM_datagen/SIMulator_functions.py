import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import io
from scipy.signal import convolve2d
import scipy.special


def PsfOtf(w, scale):
    # AIM: To generate PSF and OTF using Bessel function
    # INPUT VARIABLES
    #   w: image size
    #   scale: a parameter used to adjust PSF/OTF width
    # OUTPUT VRAIBLES
    #   yyo: system PSF
    #   OTF2dc: system OTF
    eps = np.finfo(np.float64).eps

    x = np.linspace(0, w-1, w)
    y = np.linspace(0, w-1, w)
    X, Y = np.meshgrid(x, y)

    # Generation of the PSF with Besselj.
    R = np.sqrt(np.minimum(X, np.abs(X-w))**2+np.minimum(Y, np.abs(Y-w))**2)
    yy = np.abs(2*scipy.special.jv(1, scale*R+eps) / (scale*R+eps))**2
    yy0 = fftshift(yy)

    # Generate 2D OTF.
    OTF2d = fft2(yy)
    OTF2dmax = np.max([np.abs(OTF2d)])
    OTF2d = OTF2d/OTF2dmax
    OTF2dc = np.abs(fftshift(OTF2d))

    return (yy0, OTF2dc)


def conv2(x, y, mode='same'):
    # Make it equivalent to Matlab's conv2 function
    # https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def SIMimages(opt, DIo, PSFo, OTFo):

    # AIM: to generate raw sim images
    # INPUT VARIABLES
    #   k2: illumination frequency
    #   DIo: specimen image
    #   PSFo: system PSF
    #   OTFo: system OTF
    #   UsePSF: 1 (to blur SIM images by convloving with PSF)
    #           0 (to blur SIM images by truncating its fourier content beyond OTF)
    #   NoiseLevel: percentage noise level for generating gaussian noise
    # OUTPUT VARIABLES
    #   frames:  raw sim images
    #   DIoTnoisy: noisy wide field image
    #   DIoT: noise-free wide field image

    w = DIo.shape[0]
    wo = w/2
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, w-1, w)
    [X, Y] = np.meshgrid(x, y)

    # Illuminating pattern

    # orientation direction of illumination patterns
    orientation = np.zeros(opt.Nangles)
    for i in range(opt.Nangles):
        orientation[i] = i*pi/opt.Nangles + opt.alpha + opt.angleError

    if opt.shuffleOrientations:
        np.random.shuffle(orientation)

    # illumination frequency vectors
    k2mat = np.zeros((opt.Nangles, 2))
    for i in range(opt.Nangles):
        theta = orientation[i]
        k2mat[i, :] = (opt.k2/w)*np.array([cos(theta), sin(theta)])

    # illumination phase shifts along directions with errors
    ps = np.zeros((opt.Nangles, opt.Nshifts))
    for i_a in range(opt.Nangles):
        for i_s in range(opt.Nshifts):
            ps[i_a, i_s] = 2*pi*i_s/opt.Nshifts + opt.phaseError[i_a, i_s]

    # illumination patterns
    frames = []
    for i_a in range(opt.Nangles):
        for i_s in range(opt.Nshifts):
            # illuminated signal
            sig = opt.meanInten[i_a] + opt.ampInten[i_a] * cos(2*pi*(k2mat[i_a, 0]*(X-wo) +
                          k2mat[i_a, 1]*(Y-wo))+ps[i_a, i_s])

            sup_sig = DIo*sig  # superposed signal

            # superposed (noise-free) Images
            if opt.UsePSF == 1:
                ST = conv2(sup_sig, PSFo, 'same')
            else:
                ST = np.real(ifft2(fft2(sup_sig)*fftshift(OTFo)))

            # Noise generation
            if opt.usePoissonNoise:
                # Poisson
                vals = 2 ** np.ceil(np.log2(opt.NoiseLevel)) # NoiseLevel could be 200 for Poisson: degradation seems similar to Noiselevel 20 for Gaussian
                STnoisy = np.random.poisson(ST * vals) / float(vals)
            else:
                # Gaussian
                aNoise = opt.NoiseLevel/100  # noise
                # SNR = 1/aNoise
                # SNRdb = 20*log10(1/aNoise)

                nST = np.random.normal(0, aNoise*np.std(ST, ddof=1), (w, w))
                NoiseFrac = 1  # may be set to 0 to avoid noise addition
                # noise added raw SIM images
                STnoisy = ST + NoiseFrac*nST

            frames.append(STnoisy.clip(0,1))

    return frames


def ApplyOTF(opt, Io):
    w = Io.shape[0]
    psfGT,otfGT = PsfOtf(w, 1.8*opt.scale)
    newGT = np.real(ifft2(fft2(Io)*fftshift(otfGT)))
    return newGT


# %%
def Generate_SIM_Image(opt, Io):

    w = Io.shape[0]

    # Generation of the PSF with Besselj.

    PSFo, OTFo = PsfOtf(w, opt.scale)

    DIo = Io.astype('float')

    frames = SIMimages(opt, DIo, PSFo, OTFo)

    if opt.OTF_and_GT:
        frames.append(OTFo)
        if opt.applyOTFtoGT:
            frames.append(ApplyOTF(opt,Io))
        else:
            frames.append(Io)
    stack = np.array(frames)

    # normalise
    for i in range(len(stack)):
        stack[i] = (stack[i] - np.min(stack[i])) / \
            (np.max(stack[i]) - np.min(stack[i]))

    stack = (stack * 255).astype('uint8')

    if opt.outputname is not None:
        io.imsave(opt.outputname, stack)

    return stack

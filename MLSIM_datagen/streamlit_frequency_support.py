""" ----------------------------------------
* Creation Time : Wed Jun 21 15:51:31 2023
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------"""

import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import copy

import sys
from skimage import io, transform, exposure, img_as_ubyte, img_as_float
import glob
import os
import argparse
import subprocess
from SIMulator_functions import (
    SIMimages,
    SIMimages_speckle,
    SIMimages_spots,
    PsfOtf,
    cos_wave,
    cos_wave_envelope,
    square_wave_one_third,
    square_wave,
    square_wave_large_spacing
)

from MLSIM_main import processImage

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from options import parser


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype('int')

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def GetParams_frequency_support_investigation_20230621(opt):  # uniform randomisation
    SIMopt = argparse.Namespace()

    # modulation factor
    SIMopt.ModFac = opt.ModFac


    # used to adjust PSF/OTF width
    SIMopt.PSFOTFscale = opt.PSFOTFscale
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = opt.OTF_and_GT
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination
    SIMopt.noStripes = opt.noStripes

    # function to use for stripes
    SIMopt.func = opt.func

    SIMopt.patterns = opt.patterns
    SIMopt.crop_factor = opt.crop_factor
    SIMopt.SIMmodality = opt.SIMmodality
    SIMopt.dmdMapping = opt.dmdMapping


    # normally Nframes is not set
    SIMopt.Nframes = opt.Nframes

    # --- Nframes
    if SIMopt.SIMmodality == "stripes":
        # ---- stripes
        SIMopt.Nangles = opt.Nangles

        # normally Nshifts is set like this
        # phase shifts for each stripe
        # SIMopt.Nshifts = opt.Nshifts
        SIMopt.Nshifts = SIMopt.Nframes // SIMopt.Nangles

        # number of orientations of stripes
        # orientation offset
        SIMopt.alpha = 0
        # orientation error
        SIMopt.angleError = 0
        # shuffle the order of orientations
        SIMopt.shuffleOrientations = False
        # random phase shift errors
        SIMopt.phaseError = np.zeros((SIMopt.Nangles, SIMopt.Nshifts))
        # illumination freq
        SIMopt.k2 = opt.k2

        # normally Nframes is set like this
        # SIMopt.Nframes = SIMopt.Nangles * SIMopt.Nshifts

        # amplitude of illumination intensity above mean
        SIMopt.ampInten = np.ones(SIMopt.Nangles) * SIMopt.ModFac
        # mean illumination intensity
        SIMopt.meanInten = 1 - SIMopt.ampInten
    else:
        # --- spots
        # SIMopt.Nspots = opt.Nspots
        SIMopt.spotSize = opt.spotSize

        # normally Nframes is set like this
        # SIMopt.Nframes = (SIMopt.Nspots // SIMopt.spotSize) ** 2

        # # normally not set like this
        # SIMopt.Nspots = int(np.ceil(np.sqrt(SIMopt.Nframes))) * SIMopt.spotSize
        SIMopt.Nspots = int(np.ceil(np.sqrt(SIMopt.Nframes))) * SIMopt.spotSize
        st.text(f"spotSize: {SIMopt.spotSize}, Nspots: {SIMopt.Nspots}, Nframes: {SIMopt.Nframes}")

        # amplitude of illumination intensity above mean
        SIMopt.ampInten = SIMopt.ModFac
        # mean illumination intensity
        SIMopt.meanInten = 1 - SIMopt.ampInten
        # resize amount of spots (to imitate effect of cropping from FOV on DMD/camera sensor)
        SIMopt.spotResize = 0.7 + 0.6 * (np.random.rand() - 0.5)

    SIMopt.imageSize = opt.imageSize

    return SIMopt


# Define a function to plot the OTF
def calc_otf(img):
    print(f"{img.shape}, {img.dtype}, {img.min()}, {img.max()}")
    img = img_as_float(img)
    print(f"{img.shape}, {img.dtype}, {img.min()}, {img.max()}")

    # Compute the Fourier Transform of the image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Compute the magnitude spectrum (log transformed for better visualization)
    magnitude_spectrum = np.abs(fshift)

    return magnitude_spectrum

def aggregate_frames_spots(imgstack, n):
    """
    This takes an image stack with a sparse layout of spots and combines appropriate frames to make them more dense.

    n: the nspots parameter, i.e. the number of spot locations in each grid, should be sqrt(nframes)

    returns a new image stack with twice the density and nframes/4
    """
    new_images = []

    imgstack = img_as_float(imgstack)

    for m in range(0, n//2):
        image_indices = [i for i in range(m*n, m*n + n//2)]

        new_image = None

        for i in image_indices:

            ind = []
            ind.append(i)
            ind.append(i + n//2)
            ind.append(i + n**2//2)
            ind.append(i + n**2//2 + n//2)

            # sum images corresponding to ind
            new_image = np.sum([imgstack[j] for j in ind], axis=0)

            new_images.append(new_image)

    return np.array(new_images)

def aggregate_frames_stripes(imgstack, opt):
    """
    returns a new image stack with just two phase shifts frames (equivalent to Nshifts == 2)
    """

    Nangles = opt.Nangles
    Nshifts = opt.Nshifts
    st.text(f"Nangles: {Nangles}, Nshifts: {Nshifts}")
    new_Nshifts = 2
    new_images = []

    imgstack = img_as_float(imgstack)

    for i_a in range(0, Nangles):
        for m in range(0, new_Nshifts):
            ind = [i for i in range(i_a*Nshifts + m, (i_a+1)*Nshifts, 2)]
            new_image = np.max([imgstack[j] for j in ind], axis=0)
            new_images.append(new_image)

    st.text(f"new_images.shape: {np.array(new_images).shape}")
    return np.array(new_images)

def aggregate_frames(imgstack, opt):

    if opt.SIMmodality == "stripes":
        return aggregate_frames_stripes(imgstack, opt)
    elif opt.SIMmodality == "spots":
        n = opt.Nspots
        return aggregate_frames_spots(imgstack, n)


def proj_otf(imgstack):

    ffts = []

    for img in imgstack:
        img = img_as_float(img)

        # Compute the Fourier Transform of the image
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        ffts.append(fshift)

    # max projection
    ffts = np.array(ffts)
    ffts = np.max(np.abs(ffts), axis=0)
    proj_fft = np.abs(ffts)
    return proj_fft


def estimate_noise_floor(data, percentile=5):
    return np.percentile(data, percentile)


def estimate_max(data, percentile=98):
    return np.percentile(data, percentile)

def compute_cutoff_radius(radial_profile_1d):
    """Computes the cutoff radius for given radial profile."""
    noise_floor = estimate_noise_floor(radial_profile_1d)
    max_value = estimate_max(radial_profile_1d)
    adjusted_profile = np.maximum(radial_profile_1d - noise_floor, 0)
    cutoff = max_value * 0.05
    indices = np.where(adjusted_profile < cutoff)[0]
    cutoff_radius = indices[0] if indices.size > 0 else len(radial_profile_1d) - 1
    print(f"Cutoff radius: {cutoff_radius}")

    return max_value, noise_floor, cutoff, cutoff_radius


def compute(opt, images, queue=None, progress_bar=None):
    projs = []
    wf_spectra = []

    N = len(images)

    for i in range(N):
        SIMopt = GetParams_frequency_support_investigation_20230621(opt)
        I = processImage(images[i], opt, SIMopt)

        if opt.aggregate_frames:
            I = aggregate_frames(I, SIMopt)

        wf = I.mean(axis=0)
        wf_spectrum = calc_otf(wf)
        wf_spectra.append(wf_spectrum)

        proj = proj_otf(I)
        projs.append(proj)

        if queue is not None:
            queue.put(1)

        if progress_bar is not None:
            progress_bar.progress((i + 1) / N)

    wf_spectra = np.array(wf_spectra)
    wf_spectrum = np.mean(wf_spectra, axis=0)

    projs = np.array(projs)
    proj = np.mean(projs, axis=0)
    center = (proj.shape[0] // 2, proj.shape[1] // 2)

    return I, wf, wf_spectrum, proj, center


def plot_otf_and_profile(spectrum, center):
    """Plot OTF with cutoff circle and 1D radial profile."""

    radial_profile_1d = radial_profile(spectrum, center)
    max_value, noise_floor, cutoff, cutoff_radius = compute_cutoff_radius(radial_profile_1d)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1D radial profile of the OTF
    axs[0].semilogy(radial_profile_1d)
    axs[0].axvline(
        x=cutoff_radius, color="r", linestyle="--"
    )  # Indicate the cutoff radius
    axs[0].axhline(
        y=cutoff + noise_floor, color="r", linestyle="--"
    )  # Indicate the cutoff value
    axs[0].axhline(y=noise_floor, color="r", linestyle="--")  # Indicate noise floor
    axs[0].axhline(y=max_value, color="r", linestyle="--")  # Indicate max
    axs[0].set_title("1D Radial Profile of OTF")
    axs[0].set_xlabel("Pixels along radius")

    # Add annotations
    axs[0].annotate(
        "5 % Cutoff",
        xy=(0, 1.2 * cutoff + noise_floor),
        xytext=(len(radial_profile_1d) // 2, 1.2 * cutoff + noise_floor),
    )
    axs[0].annotate(
        "Noise floor",
        xy=(0, 1.2 * noise_floor),
        xytext=(len(radial_profile_1d) // 2, 1.2 * noise_floor),
    )
    axs[0].annotate(
        "Max",
        xy=(0, 1.2 * max_value),
        xytext=(len(radial_profile_1d) // 2, 1.2 * max_value),
    )
    axs[0].annotate(
        f"Cutoff Radius={cutoff_radius}",
        xy=(1.2 * cutoff_radius, 1.5 * max_value),
        xytext=(1.2 * cutoff_radius, 1.5 * max_value),
        fontsize=8,
        rotation=90,
    )

    # Plot OTF with cutoff circle
    # normalise before plotting
    p1, p2 = noise_floor, max_value
    # plot_data = spectrum
    plot_data = np.clip((spectrum - p1) / (p2 - p1), 0, 1) + 1
    # plot_data = np.clip(spectrum, p1, p2)

    axs[1].imshow(20 * np.log(plot_data), cmap="gray")
    cutoff_circle = plt.Circle(center, cutoff_radius, color="r", fill=False)
    axs[1].add_artist(cutoff_circle)
    axs[1].set_title(f"OTF with Cutoff (radius={cutoff_radius})")

    return fig


def measure_periods(signal):
    # Get array of indices where the signal goes from 0 to 1
    cross_indices = np.where(np.diff(np.heaviside(signal, 0)) > 0)[0]

    # Calculate the differences between these indices
    periods = np.diff(cross_indices)

    return periods


# def get_base_options():
#     opt = parser.parse_args()
#     opt.imageSize = [512, 512]
#     opt.scale = 2
#     opt.sourceimages_path = "MLSIM_datagen"
#     opt.root = "MLSIM_datagen"
#     opt.out = "MLSIM_datagen"
#     opt.ModFac = 0.6
#     opt.PSFOTFscale = 0.5
#     opt.k2 = 80
#     opt.SIMmodality = "stripes"
#     opt.Nspots = 10
#     opt.Nshifts = 40
#     opt.spotSize = 2
#     opt.NoiseLevel = 15
#     opt.func = cos_wave
#     opt.OTF_and_GT = False
#     opt.patterns = 0
    # return opt

def get_wave_functions():
    return {
        "cos_wave": cos_wave,
        "cos_wave_envelope": cos_wave_envelope,
        "square_wave_one_third": square_wave_one_third,
        "square_wave": square_wave,
        "square_wave_large_spacing": square_wave_large_spacing
    }


def get_base_options():
    opt = parser.parse_args()
    opt.scale = 2
    opt.OTF_and_GT = False
    opt.patterns = 0

    # interactive options
    wave_functions = get_wave_functions()

    imageSize_x = st.sidebar.number_input('Image size, x', value=512, format="%d")
    imageSize_y = st.sidebar.number_input('Image size, y', value=512, format="%d")
    opt.imageSize = [imageSize_x, imageSize_y]

    opt.sourceimages_path = "MLSIM_datagen"
    opt.root = "MLSIM_datagen"
    opt.out = "MLSIM_datagen"

    opt.ModFac = st.sidebar.number_input('ModFac', value=0.6, format="%.2f")
    opt.PSFOTFscale = st.sidebar.number_input('PSFOTFscale', value=0.5, format="%.2f")
    opt.SIMmodality = st.sidebar.selectbox('SIM modality', options=["stripes", "spots"], index=0)
    opt.k2 = st.sidebar.number_input('[Stripes] Spatial frequency, k2', value=80, format="%d")

    opt.Nframes = st.sidebar.number_input('Frame count', value=9, format="%d")
    # opt.Nspots = st.sidebar.number_input('Number of spots', value=10, format="%d")
    # opt.Nshifts = st.sidebar.number_input('Number of shifts', value=3, format="%d")

    opt.spotSize = st.sidebar.number_input('Spot size', value=2, format="%d")
    opt.NoiseLevel = st.sidebar.number_input('Noise level', value=15, format="%d")

    func_name = st.sidebar.selectbox(
        'Pattern function', options=list(wave_functions.keys()), index=0)
    opt.func = wave_functions[func_name]

    # max number is 50
    opt.nimages = st.sidebar.number_input('Number of images', value=1, format="%d", min_value=1, max_value=50)
    opt.plot_images = st.sidebar.checkbox('Plot images', value=False)

    opt.aggregate_frames = st.sidebar.checkbox('Aggregate frames', value=False)


    return opt

def run_single(opt):
    print(opt)

    images = sorted(glob.glob("MLSIM_datagen/DIV2K_subset/*.png")[:opt.nimages])

    progress_bar = st.progress(0)
    I, wf, wf_spectrum, proj, center = compute(opt, images, progress_bar = progress_bar)

    if opt.plot_images:
        st.subheader("plot_images == True: Showing first and last frames")
        st.text(f"Stack shape: {I.shape}")
        cols = st.columns(2)
        first_frame = I[0]
        last_frame = I[1]
        with cols[0]:
            st.image(first_frame)
        with cols[1]:
            st.image(last_frame)

    st.divider()
    cols = st.columns(2)

    with cols[0]:
        fig = plt.figure(figsize=(5, 5))
        # img = io.imread(images)
        plt.imshow(wf, cmap='gray')
        plt.title('Widefield projection')
        st.pyplot(fig)
    with cols[1]:
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(I[0], cmap='gray')
        plt.title('Frame 0')
        st.pyplot(fig)


    st.pyplot(plot_otf_and_profile(wf_spectrum, center))
    st.pyplot(plot_otf_and_profile(proj, center))


def plot_images(opt, I, proj, max_value, noise_floor, cutoff, cutoff_radius, center, param1=None, param1_value=None, param2=None, param2_value=None):
    """ Helper function to plot images """
    if not opt.plot_images:
        return

    st.caption(f"Cut-off radius for {param1} = {param1_value}, {param2} = {param2_value}")
    cols = st.columns(2)
    with cols[0]:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(I[0], cmap="gray")
        st.pyplot(fig)
    with cols[1]:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        p1, p2 = noise_floor, max_value
        plot_data = np.clip((proj - p1) / (p2 - p1), 0, 1) + 1
        ax.imshow(20 * np.log(plot_data), cmap="gray")
        cutoff_circle = plt.Circle(center, cutoff_radius, color="r", fill=False)
        ax.add_artist(cutoff_circle)
        st.pyplot(fig)

def get_available_options(opt):

   return {
       # "Nshifts": "range(1, 9, 1)",
       "Nframes": "range(4, 20, 2)",
       "NoiseLevel": "range(5, 200, 20)",
       "ModFac": "np.linspace(0.1, 1.0, 10)",
       "PSFOTFscale": "np.linspace(0.1, 1.0, 10)",
       "k2": "range(30, 100, 10)",
       "SIMmodality": "['stripes', 'spots']",
       "func": list(get_wave_functions().keys()),
    }


def render_options(opt, sweep_type, param1, param1_values, param2, param2_values):

    available_options = get_available_options(opt)

    cols = st.columns(2)

    with cols[0]:
        param1 = st.selectbox('Select first parameter to vary', list(available_options.keys()), key=f"sb1_{sweep_type}")
        if sweep_type > 1:
            param2 = st.selectbox('Select second parameter to vary', list(available_options.keys()), index=1, key=f"sb2_{sweep_type}")

        with cols[1]:
            param1_values = eval(st.text_input(f'Enter values for {param1} as an expression', available_options[param1], key=f"ti1_{sweep_type}"))
            if sweep_type > 1:
                param2_values = eval(st.text_input(f'Enter values for {param2} as an expression', available_options[param2], key=f"ti2_{sweep_type}"))

    return param1, param1_values, param2, param2_values


def run_sweep(opt, sweep_type=1, param1=None, param1_values=None, param2=None, param2_values=None):
    """
    Run a parameter sweep over the given parameters
    sweep_type: 1 = 1D sweep, 2 = 2D sweep, 3 = multi-2D sweep
    """

    images = sorted(glob.glob("MLSIM_datagen/DIV2K_subset/*.png")[:opt.nimages])


    if param1 is None or param1_values is None or param2 is None or param2_values is None:
        param1, param1_values, param2, param2_values = render_options(opt, sweep_type, param1, param1_values, param2, param2_values)

    if sweep_type == 3 or st.button('Run', key=f"run_{sweep_type}"):

        progress_bar = st.progress(5)

        results_map = np.zeros((len(param1_values), len(param2_values) if param2_values else 1))

        total_evaluations = len(param1_values) * (len(param2_values) if param2_values else 1) * len(images)
        completed_evaluations = 0

        with Manager() as manager:
            queue = manager.Queue()
            with ProcessPoolExecutor() as executor:
                futures = []
                for i, param1_value in enumerate(param1_values):
                    if param2_values:
                        for j, param2_value in enumerate(param2_values):
                            opt_ = copy.deepcopy(opt)
                            setattr(opt_, param1, param1_value)
                            setattr(opt_, param2, param2_value)
                            future = executor.submit(compute, opt_, images, queue)
                            futures.append(future)
                    else:
                        opt_ = copy.deepcopy(opt)
                        setattr(opt_, param1, param1_value)
                        future = executor.submit(compute, opt_, images, queue)
                        futures.append(future)

                completed_evaluations = 0
                while completed_evaluations < total_evaluations:
                    queue.get()  # This will block until an item is available
                    completed_evaluations += 1
                    progress_bar.progress(completed_evaluations / total_evaluations)

                # Get results from futures
                for i, param1_value in enumerate(param1_values):
                    if param2_values:
                        for j, param2_value in enumerate(param2_values):
                            future = futures[i * len(param2_values) + j]
                            I, wf, wf_spectrum, proj, center = future.result()
                            radial_profile_1d = radial_profile(proj, center)
                            max_value, noise_floor, cutoff, cutoff_radius = compute_cutoff_radius(radial_profile_1d)
                            plot_images(opt, I, proj, max_value, noise_floor, cutoff, cutoff_radius, center, param1, param1_value, param2, param2_value)
                            results_map[i, j] = cutoff_radius
                    else:
                        future = futures[i]
                        I, wf, wf_spectrum, proj, center = future.result()
                        radial_profile_1d = radial_profile(proj, center)
                        max_value, noise_floor, cutoff, cutoff_radius = compute_cutoff_radius(radial_profile_1d)
                        plot_images(opt, I, proj, max_value, noise_floor, cutoff, cutoff_radius, center, param1, param1_value)
                        results_map[i, 0] = cutoff_radius

        # Display the heatmap
        if param2_values:  # 2D sweep
            fig = go.Figure(data=go.Heatmap(z=results_map,
                                 x=list(param2_values),
                                 y=list(param1_values),
                                 colorscale='viridis',
                                 colorbar=dict(title='Cut-off radius', titleside='right')))

            fig.update_layout(title=f'Cut-off radius for different {param1} and {param2} values',
                              xaxis_title=param2,
                              yaxis_title=param1,
                              )
            st.plotly_chart(fig)
        else:  # 1D sweep
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.plot(param1_values, results_map[:, 0], "o-")
            plt.title(f'Cut-off radius for different {param1} values')
            plt.xlabel(param1)
            plt.ylabel("Cut-off radius")
            st.pyplot(fig)


# Define the Streamlit app
def main():

    opt = get_base_options()

    tabs = st.tabs(['OTF Visualiser', '1D Sweep', '2D Sweep', 'Multi-2D Sweep'])

    with tabs[0]:
        st.header("Structured Illumination Microscopy (SIM) OTF Visualiser")
        run_single(opt)
    with tabs[1]:
        st.header("Cut-off Frequency Sweep")
        run_sweep(opt, 1)
    with tabs[2]:
        st.header("Cut-off Frequency 2D Sweep")
        run_sweep(opt, 2)
    with tabs[3]:
        st.header("Cut-off Frequency Multi-2D Sweeps")
        st.divider()

        available_options = get_available_options(opt)
        st.subheader("High-level variable")
        cols = st.columns(2)
        with cols[0]:
            high_level_param = st.selectbox('Select parameter to vary across 2D sweeps', list(available_options.keys()), index=5, key="sb_high")
        with cols[1]:
            high_level_values = eval(st.text_input(f'Enter values for {high_level_param} as an expression', available_options[high_level_param], key="ti_high"))

        st.divider()
        st.subheader("2D Sweep options")

        param1, param1_values, param2, param2_values = render_options(opt, 3, None, None, None, None)

        st.divider()

        if st.button('Run', key="run_3"):
            for i, high_level_value in enumerate(high_level_values):
                opt_ = copy.deepcopy(opt)
                st.subheader(f"{high_level_param}: {high_level_value}")

                if high_level_param  == "func":
                    high_level_value = get_wave_functions()[high_level_value]
                setattr(opt_, high_level_param, high_level_value)
                run_sweep(opt_, 3, param1, param1_values, param2, param2_values)

    # run_sweep(opt, 'Nshifts', list(range(1, 9, 1)), 'NoiseLevel', list(range(5, 200, 20)))
    # run_sweep(opt, 'SIMmodality', ['stripes', 'spots'], 'NoiseLevel', list(range(5, 200, 20)))



# Run the Streamlit app
if __name__ == "__main__":
    main()

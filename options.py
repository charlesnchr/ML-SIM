import argparse
import json

# training options
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="edsr", help="model to use")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument(
    "--norm", type=str, default="", help="if normalization should not be used"
)
parser.add_argument(
    "--nepoch", type=int, default=10, help="number of epochs to train for"
)
parser.add_argument(
    "--saveinterval", type=int, default=10, help="number of epochs between saves"
)
parser.add_argument("--modifyPretrainedModel", action="store_true")
parser.add_argument("--multigpu", action="store_true")
parser.add_argument("--undomulti", action="store_true")
parser.add_argument(
    "--ntrain", type=int, default=0, help="number of samples to train on"
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="",
    help="options for a scheduler, format: stepsize,gamma",
)
parser.add_argument("--log", action="store_true")
parser.add_argument(
    "--logimage", action="store_true", help="only save images in tensorboard file"
)
parser.add_argument(
    "--noise",
    type=str,
    default="",
    help="options for noise added, format: poisson,gaussVar",
)
parser.add_argument("--lambda_adv", type=float, default=0.001, help="lambda")
parser.add_argument("--lambda_pixel", type=float, default=0.01, help="lambda")
parser.add_argument("--gan_loss", type=float, default=0.03, help="gan loss percent")

# data
parser.add_argument(
    "--dataset", type=str, default="imagedataset", help="dataset to train"
)
parser.add_argument(
    "--imageSize", nargs="+", type=int, default=24, help="the low resolution image size"
)
parser.add_argument(
    "--patchSize", type=int, default=None, help="patch size for of high res output (i.e. low res is patchSize/scale)"
)
parser.add_argument("--weights", type=str, default="", help="model to retrain from")
parser.add_argument(
    "--basedir",
    type=str,
    default="",
    help="path to prepend to all others paths: root, output, weights",
)
parser.add_argument(
    "--root",
    type=str,
    default="/auto/homes/cnc39/phd/datasets",
    help="dataset to train",
)
parser.add_argument("--rootValidation", type=str, default=None, help="dataset to valid")
parser.add_argument(
    "--rootTesting", type=str, default=None, help="files for testFunction"
)
parser.add_argument(
    "--testFunction", type=str, default=None, help="extra test function to run"
)
parser.add_argument(
    "--testFunctionArgs", type=str, default=None, help="e.g. imgSize,100,nimg,5"
)
parser.add_argument(
    "--server",
    dest="root",
    action="store_const",
    const="/auto/homes/cnc39/phd/datasets",
    help="whether to use server root preset: /auto/homes/cnc39/phd/datasets",
)
parser.add_argument(
    "--local",
    dest="root",
    action="store_const",
    const="C:/phd-data/datasets/",
    help="whether to use local root preset: C:/phd-data/datasets/",
)
parser.add_argument(
    "--out", type=str, default="results", help="folder to output model training results"
)
parser.add_argument(
    "--cloud", action="store_true", help="folder to output model training results"
)
parser.add_argument("--disable_wandb", action="store_true", help="skip wandb")
parser.add_argument(
    "--disposableTrainingData",
    action="store_true",
    help="whether to delete training data after training",
)


# computation
parser.add_argument(
    "--workers", type=int, default=6, help="number of data loading workers"
)
parser.add_argument("--batchSize", type=int, default=16, help="input batch size")

# restoration options
parser.add_argument("--task", type=str, default="sr", help="restoration task")
parser.add_argument(
    "--scale", type=int, default=4, help="low to high resolution scaling factor"
)
parser.add_argument("--nch_in", type=int, default=3, help="colour channels in input")
parser.add_argument("--nch_out", type=int, default=3, help="colour channels in output")

# architecture options
parser.add_argument(
    "--narch", type=int, default=0, help="architecture-dependent parameter"
)
parser.add_argument(
    "--n_resblocks", type=int, default=10, help="number of residual blocks"
)
parser.add_argument(
    "--n_resgroups", type=int, default=10, help="number of residual groups"
)
parser.add_argument(
    "--reduction", type=int, default=16, help="number of feature maps reduction"
)
parser.add_argument("--n_feats", type=int, default=64, help="number of feature maps")
parser.add_argument(
    "--model_opts", type=json.loads, default=64, help="additional opts for model"
)

# test options
parser.add_argument(
    "--ntest",
    type=int,
    default=10,
    help="number of images to test per epoch or test run",
)
parser.add_argument(
    "--testinterval",
    type=int,
    default=1,
    help="number of epochs between tests during training",
)
parser.add_argument("--test", action="store_true")
parser.add_argument("--cpu", action="store_true")  # not supported for training
parser.add_argument(
    "--batchSize_test", type=int, default=1, help="input batch size for test loader"
)
parser.add_argument(
    "--plotinterval", type=int, default=1, help="number of epochs between plotting"
)
parser.add_argument("--nplot", type=int, default=4, help="number of plots in a test")



# ----------- ML-SIM options
parser.add_argument(
    "--sourceimages_path",
    type=str,
    default="/local/scratch/cnc39/phd/datasets/DIV2K/DIV2K_train_HR",
)
parser.add_argument(
    "--params",
    type=str,
    default="GetParams",
)
parser.add_argument(
    "--nrep", type=int, default=1, help="instances of same source image"
)
parser.add_argument("--datagen_workers", type=int, default=8, help="")
parser.add_argument("--ext", nargs="+", default=["png"])


## SIM options to control from command line

# stripes
parser.add_argument("--Nshifts", type=int, default=3)
parser.add_argument("--Nangles", type=int, default=3)
parser.add_argument("--k2", type=float, default=130.0)
parser.add_argument("--k2_err", type=float, default=15.0)

# spots
parser.add_argument("--Nspots", type=int, default=5)
parser.add_argument("--spotSize", type=int, default=1)

parser.add_argument("--PSFOTFscale", type=float, default=0.6)
parser.add_argument("--ModFac", type=float, default=0.5)
parser.add_argument("--usePSF", type=int, default=0)
parser.add_argument("--NoiseLevel", type=float, default=8)
parser.add_argument("--NoiseLevelRandFac", type=float, default=8)
parser.add_argument(
    "--phaseErrorFac", type=float, default=0.15
)  # pi/3 quite large but still feasible
parser.add_argument(
    "--alphaErrorFac", type=float, default=0.15
)  # pi/3 quite large but still feasible
parser.add_argument(
    "--angleError", type=float, default=5
)  # pi/3 quite large but still feasible
parser.add_argument("--usePoissonNoise", action="store_true")
parser.add_argument("--dontShuffleOrientations", action="store_true")
parser.add_argument("--dataonly", action="store_true")
parser.add_argument("--applyOTFtoGT", action="store_true")
parser.add_argument("--noStripes", action="store_true")
parser.add_argument("--seqSIM", action="store_true")
parser.add_argument("--skip_datagen", action="store_true")
parser.add_argument(
    "--SIMmodality",
    type=str,
    default="stripes",
    help="SIM modality",
    choices=["stripes", "spots", "speckle"],
)
parser.add_argument(
    "--patterns", action="store_true", help="Only illumination patterns"
)

# DMD options
parser.add_argument(
    "--crop_factor", action="store_true", help="Crop factor for DMD coordinates"
)
parser.add_argument(
    "--dmdMapping",
    default=0,
    type=int,
    help="""whether to map image to DMD coordinates. 0:
                    no mapping (tilted coordinate system if using oblique DMD), 1: DMD coordinate
                    transformation for acquisition, 2: predicted physical appeareance on DMD for modelling""",
    choices=[0, 1, 2],
)

# -------------

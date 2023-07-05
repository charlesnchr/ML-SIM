import sys
import numpy as np
from numpy import pi, cos, sin
import math
from skimage import io, transform
import glob
import os
import argparse
from multiprocessing import Pool
import subprocess

import run
import shutil
import wandb

from options import parser

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'MLSIM_datagen'))

from MLSIM_main import Paralleliser

if __name__ == "__main__":
    opt = parser.parse_args()

    if opt.root == "auto":
        opt.root = opt.out + "/SIMdata"
        os.makedirs(opt.root, exist_ok=True)

    print(opt)

    if not opt.skip_datagen:
        os.makedirs(opt.root, exist_ok=True)
        os.makedirs(opt.out, exist_ok=True)

        shutil.copy2("MLSIM_pipeline.py", opt.out)
        shutil.copy2("MLSIM_datagen/SIMulator_functions.py", opt.out)

        files = []
        if "imagefolder" not in opt.ext:
            for ext in opt.ext:
                files.extend(sorted(glob.glob(opt.sourceimages_path + "/*." + ext)))
        else:
            print("looking in opt", opt.sourceimages_path)
            folders = glob.glob("%s/*" % opt.sourceimages_path)
            for folder in folders:
                subfolders = glob.glob("%s/*" % folder)
                if len(subfolders) > 0:
                    if subfolders[0].endswith((".jpg", ".png")):
                        files.extend(folders)
                        break
                    files.extend(subfolders)

        if len(files) == 0:
            print("source images not found")
            sys.exit(0)
        elif (
            opt.ntrain + opt.ntest > opt.nrep * len(files)
            and opt.ntrain + opt.ntest > 0
        ):
            print(
                "ntrain + opt.ntest is too high given nrep and number of source images"
            )
            sys.exit(0)

        files = files[: math.ceil((opt.ntrain + opt.ntest) / opt.nrep)]

        if opt.ntrain + opt.ntest > 0:  # if == 0, use all
            files = files[: math.ceil((opt.ntrain + opt.ntest) / opt.nrep)]

        for file in files:
            print(file)

        Paralleliser(opt).run(files)

        print("Done generating images,", opt.root)

    # cmd = '\npython run.py ' + ' '.join(sys.argv[:])
    # print(cmd,end='\n\n')
    # subprocess.Popen(cmd,shell=True)
    if not opt.dataonly:
        if not opt.disable_wandb:
            wandb.init(project="ml-sim")
            wandb.config.update(opt)
            opt.wandb = wandb

        print("Now starting training:\n")

        run.main(opt)

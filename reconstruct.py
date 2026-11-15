"""Reconstruct raw SIM stacks with the pre-trained ML-SIM model.

    python reconstruct.py Test_data/*.tif

Each input is a TIFF stack whose first 9 frames are the raw SIM acquisition
(3 orientations x 3 phase shifts). For every stack the script writes the
ML-SIM reconstruction and the wide-field image (the mean of the raw frames)
to the output folder. The weights are downloaded on first use.
"""

import argparse
import hashlib
import os
import urllib.request

import numpy as np
import torch
from skimage import io

from models import GetModel

WEIGHTS_NAME = "DIV2K_randomised_3x3_20200317.pth"
WEIGHTS_SHA256 = "e4936f5ccf5db42009fa23ca3cbd63f53125dbd787240ca78f09e2b85c682a08"
WEIGHTS_URLS = (
    "https://github.com/charlesnchr/ML-SIM/releases/download/v1.0/" + WEIGHTS_NAME,
    "https://huggingface.co/spaces/charlesnchr/ML-SIM/resolve/main/model/" + WEIGHTS_NAME,
)


def download_weights(dest):
    """Fetch the released model, trying each mirror, and check its hash."""
    for url in WEIGHTS_URLS:
        print("Downloading pre-trained weights from", url)
        try:
            urllib.request.urlretrieve(url, dest + ".part")
        except OSError as err:
            print("  failed:", err)
            continue
        with open(dest + ".part", "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest == WEIGHTS_SHA256:
            os.replace(dest + ".part", dest)
            return
        print("  checksum mismatch, discarding the file")
        os.remove(dest + ".part")
    raise SystemExit("Could not download the weights. Fetch %s by hand and pass it with --weights." % WEIGHTS_URLS[0])


def load_model(weights, device):
    """Build the RCAN used in the paper and load the pre-trained weights."""
    opt = argparse.Namespace(
        model="rcan", task="simin_gtout", scale=1, nch_in=9, nch_out=1,
        n_resgroups=3, n_resblocks=10, n_feats=96, reduction=16, narch=0,
        cpu=True,  # build the bare network, it is moved to the device below
    )
    if not os.path.isfile(weights):
        download_weights(weights)

    checkpoint = torch.load(weights, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    net = GetModel(opt)
    net.load_state_dict(state_dict)
    return net.to(device).eval()


def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def reconstruct(net, stack, device):
    """Return (widefield, reconstruction) as float arrays in [0, 1]."""
    raw = stack[:9].astype("float32")
    widefield = minmax(raw.mean(0))
    frames = np.stack([minmax(f) for f in raw])
    with torch.no_grad():
        sr = net(torch.from_numpy(frames).unsqueeze(0).to(device))
    return widefield, sr.squeeze().clamp(0, 1).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stacks", nargs="+", help="TIFF stacks with at least 9 frames")
    p.add_argument("--weights", default=WEIGHTS_NAME, help="path to the model file")
    p.add_argument("--out", default="reconstructions", help="output folder")
    p.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is available")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    net = load_model(args.weights, device)
    os.makedirs(args.out, exist_ok=True)

    for path in args.stacks:
        stack = io.imread(path)
        if stack.ndim != 3 or stack.shape[0] < 9:
            print("Skipping %s: expected a stack of at least 9 frames, got shape %s" % (path, stack.shape))
            continue
        widefield, sr = reconstruct(net, stack, device)
        name = os.path.splitext(os.path.basename(path))[0]
        io.imsave(os.path.join(args.out, name + "_widefield.png"), (widefield * 255).astype("uint8"), check_contrast=False)
        io.imsave(os.path.join(args.out, name + "_mlsim.png"), (sr * 255).astype("uint8"), check_contrast=False)
        print("Reconstructed", path)


if __name__ == "__main__":
    main()

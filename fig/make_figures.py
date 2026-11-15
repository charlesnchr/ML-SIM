"""Regenerate the README figures from the bundled test stacks.

    python reconstruct.py Test_data/*.tif
    python fig/make_figures.py

Every image is the output of the released pre-trained model on a stack in
Test_data. Display contrast is stretched per crop, identically for the
wide-field and ML-SIM images. Needs matplotlib, pillow and ffmpeg (for the GIF).
"""

import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
BG = (13, 17, 23)
INK = (240, 246, 252)
MUTED = (139, 148, 158)

# name of stack, label, microscope, crop (x, y, size) in native pixels, display colour
SAMPLES = {
    "microtubules": ("AtheiSIM 13-43_488", "Microtubules", "interferometric SIM", (200, 150, 160), (255, 70, 200)),
    "membrane": ("AtheiSIM 14-52_488", "Cell membrane", "interferometric SIM", (120, 330, 160), (60, 200, 255)),
    "beads": ("SLM-SIM beads", "200 nm beads", "SLM-based SIM", (200, 200, 128), (255, 150, 30)),
    "er": ("SLM-SIM syrlyso640-5", "Endoplasmic reticulum", "SLM-based SIM", (200, 250, 160), (60, 230, 110)),
}


def font(size, weight="Regular"):
    for path in ("/usr/share/fonts/truetype/lato/Lato-%s.ttf" % weight,
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if weight != "Regular" else "")):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load(recon_dir, stack, kind):
    return np.asarray(Image.open(os.path.join(recon_dir, "%s_%s.png" % (stack, kind)))).astype("float32") / 255


def stretch(crop, gamma=1.0):
    lo, hi = np.percentile(crop, (0.5, 99.8))
    return np.clip((crop - lo) / (hi - lo + 1e-9), 0, 1) ** gamma


def colourise(grey, colour):
    """Black to the given colour to white, like the 'hot' lookup tables used in microscopy."""
    c = np.asarray(colour, dtype="float32") / 255
    ramp = np.clip(grey[..., None] * 1.6, 0, 1) * c
    white = np.clip((grey[..., None] - 0.55) / 0.45, 0, 1)
    return np.clip(ramp + white * (1 - ramp), 0, 1)


def panel(recon_dir, key, size, crop=None):
    """Return (widefield, mlsim) RGB crops for a sample, upscaled to size x size."""
    stack, _, _, default_crop, rgb = SAMPLES[key]
    x, y, s = crop or default_crop
    out = []
    for kind in ("widefield", "mlsim"):
        crop = stretch(load(recon_dir, stack, kind)[y:y + s, x:x + s])
        img = colourise(crop, rgb)
        img = Image.fromarray((img * 255).astype("uint8")).resize((size, size), Image.LANCZOS)
        out.append(img)
    return out


def pill(draw, xy, text, fnt, anchor="la", fill=(0, 0, 0), ink=INK):
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=fnt, anchor=anchor)
    draw.rounded_rectangle((x0 - 8, y0 - 5, x1 + 8, y1 + 5), radius=6, fill=fill)
    draw.text(xy, text, font=fnt, anchor=anchor, fill=ink)


def hero_frame(panels, keys, p, size, gap, margin, caption_h):
    """One frame of the wipe animation. p in [0, 1] is the fraction showing ML-SIM."""
    width = 2 * margin + len(keys) * size + (len(keys) - 1) * gap
    canvas = Image.new("RGB", (width, margin + size + caption_h), BG)
    draw = ImageDraw.Draw(canvas)
    label, caption, small = font(15, "Bold"), font(17, "Bold"), font(14)
    for i, key in enumerate(keys):
        wf, sr = panels[key]
        x0 = margin + i * (size + gap)
        split = int(round(p * size))
        tile = wf.copy()
        tile.paste(sr.crop((0, 0, split, size)), (0, 0))
        td = ImageDraw.Draw(tile)
        if split > 86:
            pill(td, (12, 12), "ML-SIM", label)
        if size - split > 116:
            pill(td, (size - 12, 12), "WIDE-FIELD", label, anchor="ra")
        if 0 < split < size:
            td.line((split, 0, split, size), fill=INK, width=2)
        canvas.paste(tile, (x0, margin))
        _, name, scope, _, _ = SAMPLES[key]
        draw.text((x0, margin + size + 12), name, font=caption, fill=INK)
        draw.text((x0 + size, margin + size + 14), scope, font=small, fill=MUTED, anchor="ra")
    return canvas


def make_hero(recon_dir, out, keys=("microtubules", "beads", "er"), size=300, fps=20):
    panels = {k: panel(recon_dir, k, size) for k in keys}
    ease = lambda n: [(1 - np.cos(np.pi * t)) / 2 for t in np.linspace(0, 1, n)]
    sweep = ease(int(fps * 2.0))
    positions = [0.0] * int(fps * 0.9) + sweep + [1.0] * int(fps * 1.6) + sweep[::-1]
    tmp = tempfile.mkdtemp()
    try:
        for i, p in enumerate(positions):
            hero_frame(panels, keys, p, size, 12, 16, 46).save(os.path.join(tmp, "f%04d.png" % i))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
             "-vf", "split[a][b];[a]palettegen=max_colors=160:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
             out], check=True)
    finally:
        shutil.rmtree(tmp)


def make_grid(recon_dir, out, keys=("microtubules", "membrane", "beads", "er"), size=380):
    gap, margin, left, head = 10, 20, 44, 78
    width = left + margin + len(keys) * size + (len(keys) - 1) * gap + margin
    height = head + 2 * size + gap + margin
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    for i, key in enumerate(keys):
        x0 = left + margin + i * (size + gap)
        _, name, scope, _, _ = SAMPLES[key]
        draw.text((x0, 12), name, font=font(26, "Bold"), fill=INK)
        draw.text((x0, 46), scope, font=font(18), fill=MUTED)
        for row, img in enumerate(panel(recon_dir, key, size, GRID_CROPS.get(key))):
            canvas.paste(img, (x0, head + row * (size + gap)))
    for row, text in enumerate(("WIDE-FIELD", "ML-SIM")):
        strip = Image.new("RGB", (size, 36), BG)
        ImageDraw.Draw(strip).text((size // 2, 18), text, font=font(23, "Bold"),
                                   fill=INK if row else MUTED, anchor="mm")
        canvas.paste(strip.rotate(90, expand=True), (margin - 2, head + row * (size + gap)))
    canvas.save(out, optimize=True)


def make_social(recon_dir, out):
    """1280x640 image for the repository's social preview."""
    w, h, size = 1280, 640, 300
    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 52), "ML-SIM", font=font(84, "Black"), fill=INK)
    draw.text((66, 156), "Super-resolution microscopy reconstruction, trained on simulation alone",
              font=font(31, "Medium"), fill=MUTED)
    draw.text((w - 64, 100), "Biomedical Optics Express 2021", font=font(22), fill=MUTED, anchor="ra")
    keys = ("microtubules", "beads", "er")
    gap = 26
    x = (w - (len(keys) * size + (len(keys) - 1) * gap)) // 2
    for key in keys:
        wf, sr = panel(recon_dir, key, size)
        tile = wf.copy()
        tile.paste(sr.crop((size // 2, 0, size, size)), (size // 2, 0))
        td = ImageDraw.Draw(tile)
        td.line((size // 2, 0, size // 2, size), fill=INK, width=2)
        pill(td, (12, 12), "WIDE-FIELD", font(14, "Bold"))
        pill(td, (size - 12, 12), "ML-SIM", font(14, "Bold"), anchor="ra")
        canvas.paste(tile, (x, 236))
        draw.text((x, 236 + size + 14), SAMPLES[key][1], font=font(22, "Bold"), fill=INK)
        x += size + gap
    canvas.save(out, optimize=True)


def make_profile(recon_dir, out):
    """Line profile across neighbouring beads, wide-field against ML-SIM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.ndimage import map_coordinates

    for weight in ("Regular", "Bold"):
        path = "/usr/share/fonts/truetype/lato/Lato-%s.ttf" % weight
        if os.path.isfile(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = "Lato"

    stack = SAMPLES["beads"][0]
    wf, sr = load(recon_dir, stack, "widefield"), load(recon_dir, stack, "mlsim")
    (x0, y0), (x1, y1) = PROFILE_LINE
    n = int(np.hypot(x1 - x0, y1 - y0) * 4)
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    dist = np.linspace(0, np.hypot(x1 - x0, y1 - y0), n)
    prof = [map_coordinates(im, [ys, xs], order=3) for im in (wf, sr)]
    prof = [(p - p.min()) / (p.max() - p.min()) for p in prof]

    bg, ink, muted = [tuple(v / 255 for v in c) for c in (BG, INK, MUTED)]
    blue, orange = "#3987e5", "#d95926"
    grey = lambda im: np.repeat(im[..., None], 3, axis=2)
    cx, cy, half = PROFILE_CROP
    fig = plt.figure(figsize=(11, 3.7), dpi=160, facecolor=bg)
    side = 0.69 * 3.7 / 11  # square image panels
    boxes = [(0.012, 0.2, side, 0.69), (0.03 + side, 0.2, side, 0.69), (0.56, 0.2, 0.425, 0.69)]
    for i, (im, title) in enumerate(((wf, "WIDE-FIELD"), (sr, "ML-SIM"))):
        ax = fig.add_axes(boxes[i])
        crop = stretch(im[cy - half:cy + half, cx - half:cx + half])
        ax.imshow(grey(crop), extent=(cx - half, cx + half, cy + half, cy - half),
                  interpolation="lanczos")
        ax.plot([x0, x1], [y0, y1], color=blue if i else orange, lw=1.6, dashes=(4, 3))
        ax.set_title(title, color=ink, fontsize=15, fontweight="bold", pad=8)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax = fig.add_axes(boxes[2], facecolor=bg)
    ax.plot(dist, prof[0], color=orange, lw=2, label="Wide-field")
    ax.plot(dist, prof[1], color=blue, lw=2, label="ML-SIM")
    ax.text(2.9, 0.62, "Wide-field", color=ink, ha="right", va="center", fontsize=15, fontweight="bold")
    peak = int(np.argmax(prof[1]))
    ax.annotate("ML-SIM", (dist[peak], prof[1][peak]), xytext=(6, 2), textcoords="offset points",
                color=ink, fontsize=15, fontweight="bold")
    ax.set_xlabel("Distance along the dashed line (pixels)", color=muted, fontsize=14)
    ax.set_ylabel("Normalised intensity", color=muted, fontsize=14)
    ax.set_ylim(-0.03, 1.12), ax.set_xlim(dist[0], dist[-1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(range(0, 18, 2))
    ax.tick_params(colors=muted, labelsize=13, length=0)
    ax.grid(axis="y", color=muted, alpha=0.18, lw=0.8)
    for side, s in ax.spines.items():
        s.set_visible(side == "bottom")
        s.set_color(muted)
    fig.savefig(out, facecolor=bg)
    plt.close(fig)


# crops for the results grid, different from the ones in the animation
GRID_CROPS = {"microtubules": (60, 60, 160), "beads": (40, 40, 128), "er": (120, 100, 160)}

# line through a row of three beads in the SLM-SIM bead sample, native pixel coordinates,
# and the (centre x, centre y, half width) of the region shown next to the plot
PROFILE_LINE = ((397.6, 132.3), (414.8, 128.0))
PROFILE_CROP = (406, 129, 14)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recon", default="reconstructions", help="folder written by reconstruct.py")
    p.add_argument("--out", default=HERE)
    args = p.parse_args()
    make_hero(args.recon, os.path.join(args.out, "hero.gif"))
    make_grid(args.recon, os.path.join(args.out, "results.png"))
    make_social(args.recon, os.path.join(args.out, "social-preview.png"))
    if PROFILE_CROP[2]:
        make_profile(args.recon, os.path.join(args.out, "line-profile.png"))

"""

"""
### MODULES

import os
import argparse
from pathlib import Path
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

### FUNCTIONS
def plot_tiff(
    path: str,
    save_dir: str = "C:/Users/EduAm/Desktop/Escritorio/Lund/Uni/Research Project 2/",
    cmap_list=None,
    save_channels: bool = False,
    dpi: int = 300,
    normalization: str = "minmax",  # 'minmax', 'percentile', or 'global'
    brightness_factors=None,        # list of floats (one per channel)
    percentiles: tuple = (1, 99)    # used for percentile normalization
):
    """
    Multiplex TIFF image visualizer with flexible normalization and coloring.

    Parameters
    ----------
    path : str
        Path to the multi-channel TIFF image.
    save_dir : str
        Directory where output PNGs will be saved.
    cmap_list : list
        List of colors (hex, named, or RGB tuples). Default = preset 6-plex colors.
    save_channels : bool
        Whether to save each channel as grayscale PNG.
    dpi : int
        Resolution for saved images.
    normalization : str
        One of ['minmax', 'percentile', 'global'].
    brightness_factors : list
        Per-channel brightness scaling factors. In case one needs to be brighter than the others
    percentiles : tuple
        (low, high) percentiles for 'percentile' normalization.
    """

    # Load TIFF file
    img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[0] < img.shape[-1]:
        channels, h, w = img.shape
    elif img.ndim == 3 and img.shape[-1] < img.shape[0]:
        h, w, channels = img.shape
        img = np.moveaxis(img, -1, 0)
    else:
        raise ValueError("Unexpected image shape, must be (C, H, W) or (H, W, C)")

    os.makedirs(save_dir, exist_ok=True)

    # Set up color palette
    default_colors = [
        "#0000FF",  # DAPI (blue)
        "#FF00FF",  # FOXP3 (magenta)
        "#00FF00",  # PanCK (green)
        "#FF0000",  # CD8 (red)
        "#FFFF00",  # CD163 (yellow)
        "#00FFFF",  # α-SMA (cyan)
    ]
    if cmap_list is None:
        cmap_list = default_colors[:channels]

    # Convert all colors to RGB tuples (0–1 range)
    cmap_list = [to_rgb(c) for c in cmap_list]

    # Normalization functions 
    def normalize_minmax(x):
        x = x.astype(float)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def normalize_percentile(x, p_low=1, p_high=99):
        lo, hi = np.percentile(x, (p_low, p_high))
        x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
        return x

    def normalize_global(x, vmin, vmax):
        return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Choose normalization mode 
    if normalization == "global":
        vmin, vmax = img.min(), img.max()

    # Brightness factors check
    if brightness_factors is None:
        brightness_factors = [1.0] * channels
    elif len(brightness_factors) != channels:
        raise ValueError("brightness_factors must have one value per channel")

    # Build composite png image

    composite = np.zeros((h, w, 3))
    for i in range(channels):
        if normalization == "minmax":
            ch_norm = normalize_minmax(img[i])
        elif normalization == "percentile":
            ch_norm = normalize_percentile(img[i], *percentiles)
        elif normalization == "global":
            ch_norm = normalize_global(img[i], vmin, vmax)
        else:
            raise ValueError("Invalid normalization method. Choose 'minmax', 'percentile', or 'global'")

        color = np.array(cmap_list[i])
        weight = brightness_factors[i]
        composite += color * (ch_norm[..., None] * weight)

        # Optional: save each channel
        if save_channels:
            plt.imshow(ch_norm, cmap="gray")
            plt.axis("off")
            plt.title(f"Channel {i}")
            plt.savefig(os.path.join(save_dir, f"channel_{i}.png"),
                        bbox_inches="tight", dpi=dpi)
            plt.close()

    composite = np.clip(composite, 0, 1)

    # Save composite png image
    plt.figure(figsize=(8, 8))
    plt.imshow(composite)
    plt.axis("off")
    plt.title(f"Composite Overlay ({normalization} normalization)")
    out_path = os.path.join(save_dir, os.path.basename(path).replace(".tiff", f"_{normalization}_composite.png"))
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()
### INPUT CHECK
parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", help= "", required=True)

args = parser.parse_args()

input_path = Path(args.input_folder)

if input_path.exists() and input_path.is_dir():
### MAIN

# Given an input folder, parse it completely (even inside subfolders) and apply to every tiff file found there the plotting function
# Catch the error and keep going even if one fails

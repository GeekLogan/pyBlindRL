#   -------------------------------------------------------------
#   Copyright (c) Logan Walker. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""A Python implementation of blind Richardson-Lucy deconvolution"""
from __future__ import annotations

import numpy as np
import torch

__version__ = "0.0.2"


def gaussian_3d(shape, center=None, sigma=None):
    """
    Generate a 3D Gaussian array.

    Parameters:
        shape (tuple): Shape of the output array (depth, height, width).
        center (tuple, optional): Center of the Gaussian in the array. Defaults to the center of the array.
        sigma (tuple, optional): Standard deviation of the Gaussian in each direction.
                                 Defaults to half of the shape in each direction.

    Returns:
        ndarray: 3D Gaussian array.
    """
    if center is None:
        center = tuple(dim // 2 for dim in shape)
    if sigma is None:
        sigma = tuple(dim / 2 for dim in shape)

    grid = np.ogrid[[slice(0, s) for s in shape]]
    distances = [(grid[axis] - center[axis]) ** 2 / (2 * sigma[axis] ** 2) for axis in range(3)]
    gaussian_array = np.exp(-sum(distances))

    gaussian_array -= gaussian_array.min()
    gaussian_array /= gaussian_array.max()

    return gaussian_array


otf = np.zeros_like(img, dtype=np.complex128)
otf = np.ones_like(img, dtype=np.complex128)

otf[
    int(img.shape[0] / 2 - psf.shape[0] / 2) :,
    int(img.shape[1] / 2 - psf.shape[1] / 2) :,
    int(img.shape[2] / 2 - psf.shape[2] / 2) :,
][: psf.shape[0], : psf.shape[1], : psf.shape[2]] += psf

otf += 1e-8

print(otf.min(), otf.max())

for axis, axis_size in enumerate(img.shape):
    otf = np.roll(otf, -int(axis_size / 2), axis=axis)

# otf = np.fft.fftn( otf )


def RL_deconv(image, otf, iterations, target_device="cpu", eps=1e-10):
    with torch.no_grad():
        out = torch.clone(image).detach().to(target_device)

        depth, height, width = out.shape
        window = 25
        masks = [
            (slice(0, window), slice(0, window), slice(0, window)),  # Top left corner
            (slice(0, window), slice(0, window), slice(width - window, width)),  # Top right corner
            (slice(0, window), slice(height - window, height), slice(0, window)),  # Bottom left corner
            (slice(0, window), slice(height - window, height), slice(width - window, width)),  # Bottom right corner
            (slice(depth - window, depth), slice(0, window), slice(0, window)),  # Front top left corner
            (slice(depth - window, depth), slice(0, window), slice(width - window, width)),  # Front top right corner
            (
                slice(depth - window, depth),
                slice(height - window, height),
                slice(0, window),
            ),  # Front bottom left corner
            (
                slice(depth - window, depth),
                slice(height - window, height),
                slice(width - window, width),
            ),  # Front bottom right corner
        ]

        for _ in range(iterations):
            tmp = torch.fft.fftn(out)

            # tmp *= otf
            for mask in masks:
                tmp[mask] *= otf[mask]

            tmp = torch.fft.ifftn(tmp)

            tmp += eps  # prevent 0-division
            tmp = image / tmp

            tmp = torch.fft.fftn(tmp)
            # tmp *= otf.conj()
            for mask in masks:
                tmp[mask] *= otf[mask].conj()
            tmp = torch.fft.ifftn(tmp)

            out *= tmp

        return out


def RL_deconv_otf(image, otf, iterations, rl_iter=10, target_device="cpu"):
    with torch.no_grad():
        out = torch.clone(image).detach().to(target_device)
        out_psf = torch.clone(otf).detach().to(target_device)

        for _bld in tqdm.trange(iterations):
            out = torch.fft.fftn(out)
            for _ in range(rl_iter):
                tmp = torch.fft.fftn(out_psf)
                tmp *= out
                tmp = torch.fft.ifftn(tmp)
                tmp += 1e-9
                tmp = image / tmp

                tmp = torch.fft.fftn(tmp)
                tmp *= out.conj()
                tmp = torch.fft.ifftn(tmp)

                out_psf *= tmp

                del tmp
            out = torch.fft.ifftn(out)

            out_psf = torch.fft.fftn(out_psf)
            for _ in range(rl_iter):
                tmp = torch.fft.fftn(out)
                tmp *= out_psf
                tmp = torch.fft.ifftn(tmp)
                tmp += 1e-9
                tmp = image / tmp

                tmp = torch.fft.fftn(tmp)
                tmp *= out_psf.conj()
                tmp = torch.fft.ifftn(tmp)

                out *= tmp
                out += 0.01 * image

                del tmp
            out_psf = torch.fft.ifftn(out_psf)

        oout = torch.abs(out).to("cpu").numpy().astype(float)
        oout_psf = torch.abs(out_psf).to("cpu").numpy().astype(float)

        del out, out_psf

        return oout, oout_psf

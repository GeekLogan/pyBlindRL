#   -------------------------------------------------------------
#   Copyright (c) Logan Walker. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

import numpy as np
import tifffile as tiff
import torch
import tqdm


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

def emplace_center(img, img2, dtype=np.complex128):
    """
    Place given img2 into the center of a new array given dimensions of img

    Parameters:
        img (3d numpy array): Image to use as a template
        img2 (3d numpy array): Image to place
    """
    out = np.zeros_like(img, dtype=dtype)

    out[
        int(img.shape[0] / 2 - img2.shape[0] / 2) :,
        int(img.shape[1] / 2 - img2.shape[1] / 2) :,
        int(img.shape[2] / 2 - img2.shape[2] / 2) :,
    ][: img2.shape[0], : img2.shape[1], : img2.shape[2]] += img2

    return out


def generate_initial_psf(img):
    """
    Creates a PSF image based on a Gaussian centered on the corners

    Parameters:
        img (3d numpy array): Image to use as a template
    """
    psf_shape = (64, 64, 64)
    psf = gaussian_3d(psf_shape, sigma=(1, 1, 2))

    out = emplace_center(img, psf)
    out += 1

    out = roll_psf(out)

    return out

def generate_initial_psf_smaller(img, psf_shape):
    psf = gaussian_3d(psf_shape, sigma=(1, 1, 2))

    out = emplace_center(img, psf)
    out += 1

    out = roll_psf(out)

    return out


def roll_psf(img):
    """
    Roll PSF at center of image to edge of image.

    Parameters:
        img (3d numpy array): Image to roll
    """

    for axis, axis_size in enumerate(img.shape):
        img = np.roll(img, -int(axis_size / 2), axis=axis)

    return img


def unroll_psf(img):
    """
    Move PSF aligned with corners and roll to center.

    Parameters:
        img (3d numpy array): Image to unroll
    """

    for axis, axis_size in enumerate(img.shape):
        img = np.roll(img, int(axis_size / 2), axis=axis)

    return img


def normalize_psf(psf):
    """
    Renormalized the given PSF function between 0...1

    Parameters:
        psf (3d numpy array): PSF to normalize
    """

    out = np.zeros(psf.shape, dtype=np.float32)
    out[...] += psf.astype(np.float32)
    out -= out.min()
    out /= out.max()
    return out


def clip_psf(psf, shape=(64, 64, 64)):
    """
    Clip out the center of a PSF image (unrolled)

    Parameter:
        psf (3d numpy array): Input image
        shape (3-tuple, int): Size to clip
    """

    out = psf[tuple(slice((a // 2) - (b // 2), None, None) for a, b in zip(psf.shape, shape))][
        tuple(slice(None, a, None) for a in shape)
    ]

    return np.copy(out)


def intensity_match_image(img, img_deconv, method="peak"):
    """
    Match the intensity of img_deconv to the input image img

    Parameters:
        img (3d numpy array):
        img_deconv (3d numpy array):
        method (str): name of the method to use
    """

    if method == "peak":
        out = np.copy(img_deconv)

        out = out.astype(np.float64)

        out /= out.max()
        out *= img.max()

        return out.astype(np.uint16)


def RL_deconv(image, out, otf, iterations, target_device="cpu", eps=1e-10, approx=True):
    """
    Perform unblinded RL deconvolution

    Parameters:
        img (3d numpy array): Image to deconvolute
        otf (3d numpy array): OTF to deconvolute with
        iterations (int): number of iterations to perform
        target_device (str): torch device to creat output on
        eps (float): value added to prevent zero-division error
        approx (bool): flag to enable fast approximation optimizations
    """

    with torch.no_grad():


        start_mem = torch.cuda.memory_allocated(target_device)

        image = torch.clone(image).detach().to(target_device)

        out = torch.clone(out).detach().to(target_device)

        otf = torch.clone(otf).detach().to(target_device)

        # print("Deconvolution Memory Usage (Bytes)")
        end_mem = 0

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
            end_mem = torch.cuda.memory_allocated(target_device)

            if approx:
                for mask in masks:
                    tmp[mask] *= otf[mask]
            else:
                tmp *= otf

            tmp = torch.fft.ifftn(tmp)

            tmp += eps  # prevent 0-division
            tmp = image / tmp

            tmp = torch.fft.fftn(tmp)
            if approx:
                for mask in masks:
                    tmp[mask] *= otf[mask].conj()
            else:
                tmp *= otf.conj()
            tmp = torch.fft.ifftn(tmp)

            out *= tmp

        out = torch.abs(out).cpu().numpy().astype(float)
        return out, end_mem

def RL_deconv_2D(image, out, otf, iterations, target_device="cpu", eps=1e-10):
    """
    Perform unblinded RL deconvolution

    Parameters:
        img (2d numpy array): Image to deconvolute
        otf (2d numpy array): OTF to deconvolute with
        iterations (int): number of iterations to perform
        target_device (str): torch device to creat output on
        eps (float): value added to prevent zero-division error
    """

    with torch.no_grad():


        start_mem = torch.cuda.memory_allocated(target_device)

        image = torch.clone(image).detach().to(target_device)

        out = torch.clone(out).detach().to(target_device)

        otf = torch.clone(otf).detach().to(target_device)

        end_mem = torch.cuda.memory_allocated(target_device)

        # print("Deconvolution Memory Usage (Bytes)")
        # print(end_mem)

        for _ in range(iterations):
            tmp = torch.fft.fftn(out)

            tmp *= otf

            tmp = torch.fft.ifftn(tmp)

            tmp += eps  # prevent 0-division
            tmp = image / tmp

            tmp = torch.fft.fftn(tmp)

            tmp *= otf.conj()
            tmp = torch.fft.ifftn(tmp)

            out *= tmp

        out = torch.abs(out).cpu().numpy().astype(float)
        return out, end_mem


def RL_deconv_blind(gt_image, out_image, psf, iterations=20, rl_iter=10, eps=1e-9, reg_factor=0.01, target_device="cpu"):
    """
    Perform Blinded RL deconvolution

    Parameters:
        image (3d tensor): Image to deconvolute
        psf (3d tensor): Guess PSF to start deconvolution with
        iterations (int): number of iterations to perform
        rl_iter (int): number of sub-iterations to perform
        target_device (str): torch device to creat output on
        eps (float): value added to prevent zero-division error
        reg_factor (float): value used to regularize the image
        target_device (str): name of pytorch device to use for calculation
    """

    start_memory = torch.cuda.memory_allocated(target_device)

    with torch.no_grad():
        tmp_image = gt_image.to(target_device)
        out = torch.clone(out_image).detach().to(target_device)
        out_psf = torch.clone(psf).detach().to(target_device)

        # print("Blinded Memory Usage (Bytes)")
        # print(end_memory)
        end_memory = 0

        for _bld in tqdm.trange(iterations):
            out = torch.fft.fftn(out)
            for _ in range(rl_iter):
                tmp = torch.fft.fftn(out_psf)
                end_memory = torch.cuda.memory_allocated(target_device)
                tmp *= out
                tmp = torch.fft.ifftn(tmp)
                tmp += eps
                tmp = tmp_image / tmp

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
                tmp += eps
                tmp = tmp_image / tmp

                tmp = torch.fft.fftn(tmp)
                tmp *= out_psf.conj()
                tmp = torch.fft.ifftn(tmp)

                out *= tmp
                out += reg_factor * tmp_image

                del tmp
            out_psf = torch.fft.ifftn(out_psf)

        oout = torch.abs(out.cpu()).numpy().astype(float)
        oout_psf = torch.abs(out_psf.cpu()).numpy().astype(float)

        del out, out_psf, tmp_image

        return oout, oout_psf, end_memory

#Takes in the array of overlapping image tiles and compute some average from the
#edges of the images

## Just do the deconvolution with tiles sizes of ~150 x 150 then use a weighted averaging to
#hopefully remove the edge effects
def edge_correction(image_array, large_image_width, target_image_width, img_size):

    overlap = int((large_image_width - target_image_width) / 2)

    gradient = np.zeros((img_size[0], large_image_width, large_image_width))
    regions = [large_image_width / 6, 2 * large_image_width / 6, 3 * large_image_width / 6, 4 * large_image_width / 6, 5 * large_image_width / 6, large_image_width]

    for i in range(large_image_width):
        if i <= regions[0]:
            gradient[:, i, :] = (i / regions[0]) * 50 + 0
        elif i <= regions[1]:
            gradient[:, i, :] = ((i - regions[0]) / regions[0]) * 50 + 50
        elif i <= regions[3]:
            gradient[:, i, :] = 100
        elif i <= regions[4]:
            gradient[:, i, :] = (-(i - regions[3]) / regions[0]) * 50 + 100
        else:
            gradient[:, i, :] = (-(i - regions[4]) / regions[0]) * 50 + 50
    for i in range(large_image_width):
        for j in range(large_image_width):
            if j <= regions[0]:
                val = min(gradient[0, i, j], (j / regions[0]) * 50 + 0)

                gradient[:, i, j] = val
            elif j <= regions[1]:
                val = min(gradient[0, i, j], ((j - regions[0]) / regions[0]) * 50 + 50)

                gradient[:, i, j] = val
            elif j <= regions[3]:
                val = min(gradient[0, i, j], 100)

                gradient[:, i, j] = val
            elif j <= regions[4]:
                val = min(gradient[0, i, j], (-(j - regions[3]) / regions[0]) * 50 + 100)

                gradient[:, i, j] = val
            else:
                val = min(gradient[0, i, j], (-(j - regions[4]) / regions[0]) * 50 + 50)

                gradient[:, i, j] = val


    tiff.imwrite("./gradient.tiff", gradient)

    gradient = gradient / 100

    output = np.zeros(img_size)
    counts = np.zeros(img_size)

    tile_num = img_size[1] / target_image_width

    for i in range(int(tile_num)):
        for j in range(int(tile_num)):
            if i > 0 and i < tile_num - 1 and j > 0 and j < tile_num - 1:
                output[:, (i*target_image_width) - overlap:((i+1) * target_image_width) + overlap, (j*target_image_width) - overlap:((j+1) * target_image_width) + overlap] = output[:, (i*target_image_width) - overlap:((i+1) * target_image_width) + overlap, (j*target_image_width) - overlap:((j+1) * target_image_width) + overlap] + (image_array[i, j, :, :, :] * gradient)
                counts[:, (i*target_image_width) - overlap:((i+1) * target_image_width) + overlap, (j*target_image_width) - overlap:((j+1) * target_image_width) + overlap] = counts[:, (i*target_image_width) - overlap:((i+1) * target_image_width) + overlap, (j*target_image_width) - overlap:((j+1) * target_image_width) + overlap] + gradient
            else:
                counts[:, (i*target_image_width):((i+1) * target_image_width), (j*target_image_width):((j+1) * target_image_width)] = 1

    for i in range(img_size[1]):
        for j in range(img_size[2]):
            if counts[0, i, j] != 1 and counts[0, i, j] > 0:
                output[:, i, j] = output[:, i, j] * (1 / counts[0, i, j])

    return output

def slice_blending(image_array, img_size):

    blended = np.copy(image_array)

    for i in range(img_size[0]):
        if i == 0:
            blended[i, :, :] = image_array[i, :, :] * 0.5 + image_array[i + 1, :, :] * 0.5
        if i == img_size[0] - 1:
            blended[i, :, :] = image_array[i, :, :] * 0.5 + image_array[i - 1, :, :] * 0.5
        if i > 0 and i < img_size[0] - 1:
            blended[i, :, :] = image_array[i, :, :] * 0.5 + image_array[i - 1, :, :] * 0.25 + image_array[i + 1, :, :] * 0.25

    return blended
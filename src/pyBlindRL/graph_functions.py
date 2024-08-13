import scipy.signal
from commands import generate_initial_psf, RL_deconv_blind, unroll_psf, clip_psf, normalize_psf
from utility import clear_dir
import scipy
import cv2
import matplotlib.pyplot as plt
import time
import tifffile as tiff
import torch
import os
from cloudvolume import CloudVolume
import random
from tqdm import tqdm
import numpy as np
import glob

def graph_functions(function_dir, output_file):
    files = glob.glob(function_dir + "/*.tiff")
    files.sort()

    print(len(files))

    f = plt.figure()

    for i in tqdm(range(len(files))):
        function = tiff.imread(files[i])
        function.astype(np.uint16)

        function = normalize_psf(function)

        print(function.shape)

        f.set_size_inches(15, 5)

        line = function[32, 32, 22:42]

        dist = np.arange(0, line.shape[0], 1)

        dist = dist - line.shape[0] / 2

        plt.plot(dist, line)

    plt.ylabel("PSF Normalized Value")
    plt.xlabel("Dist from Center")


    plt.savefig(output_file)
    plt.close()


def graph_fwhm(function_dir, output_file):
    files = glob.glob(function_dir + "/*.tiff")
    files.sort()

    print(len(files))

    f = plt.figure()

    fwhms = []
    iters = []

    for i in tqdm(range(len(files))):
        function = tiff.imread(files[i])
        function.astype(np.uint16)

        function = normalize_psf(function)

        line = function[32, 32, :]

        print(line.shape)

        peaks, _ = scipy.signal.find_peaks(line, height= 1)

        widths, _, _, _ = scipy.signal.peak_widths(line, peaks, 0.5)

        fwhms.append(widths[0])
        iters.append((i + 1))


    plt.plot(iters, fwhms)
    plt.ylabel("Full Width at Half Maximum")
    plt.xlabel("Iteration")


    plt.savefig(output_file)
    plt.close()

graph_fwhm("/mnt/turbo/jfeggerd/outputs_rolling_edge/functions", "./fwhms.png")








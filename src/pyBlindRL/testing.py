from commands import generate_initial_psf, RL_deconv_blind
from utility import clear_dir
import scipy
import cv2
import time
import tifffile as tiff
import torch
import os
from cloudvolume import CloudVolume
import random
from tqdm import tqdm
import numpy as np
import glob


output_dir = "/mnt/c/users/jfegg/documents/cai_lab/pyBlindRL/outputs"
blurred_dir = output_dir + "/blurred"
deconv_dir = output_dir + "/deconv"
output_functions_dir = output_dir + "/functions"
imgs_dir = output_dir + "/imgs"
initial_functions_dir = output_dir + "/initial_functions"

device = torch.device("cuda", 0)


x_lim = 28000
y_lim = 28000
count = 0

def deconvolve_cloud(x, y, z, xy_size, slices, section_size, blur_radius, device, output_dir):
    vol = CloudVolume('precomputed://https://ntracer2.cai-lab.org/data2/051524_bitbow_ch0', parallel=True, progress=True)

    img = vol[x:x+xy_size, y:y+xy_size, z:z+slices]

    img = img [:, :, :, 0]

    img = img.transpose(2, 0, 1)

    blurred_dir = output_dir + "/blurred"
    deconv_dir = output_dir + "/deconv"
    output_functions_dir = output_dir + "/functions"
    imgs_dir = output_dir + "/imgs"
    initial_functions_dir = output_dir + "/initial_functions"

    dirs = [blurred_dir, deconv_dir, output_functions_dir, imgs_dir, initial_functions_dir]

    for i in dirs:
        clear_dir(i)

    img_tensor = torch.from_numpy(np.array(img).astype(np.int16))

    blurred = scipy.ndimage.gaussian_filter(img, (1, 1, 2), radius = blur_radius)


    output_img = np.zeros((img_tensor.shape))

    num = 0

    for i in tqdm(range(int(xy_size / section_size))):
        for j in range(int(xy_size / section_size)):
            blurred_section = torch.clone(torch.from_numpy((blurred[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)]).astype(np.int16)))

            psf_guess = generate_initial_psf(blurred_section)

            output, output_psf = RL_deconv_blind(blurred_section, torch.from_numpy(psf_guess), target_device=device, iterations=20)

            output_img[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = output

            if i == 0 and j == 0:
                tiff.imwrite(imgs_dir + "/img_" + str(num) + ".tiff", img_tensor.detach().cpu().numpy())
                tiff.imwrite(blurred_dir + "/img_" + str(num) + ".tiff", blurred)
                tiff.imwrite(output_functions_dir + "/img_" + str(num) + ".tiff", output_psf)
                tiff.imwrite(initial_functions_dir + "/img_" + str(num) + ".tiff", psf_guess)

            if i == int(xy_size / section_size) and j == int(xy_size / section_size):
                tiff.imwrite(deconv_dir + "/img_" + str(num) + ".tiff", output_img.astype(np.uint16))

            num += 1

seconds = time.time()

deconvolve_cloud(7000, 10000, 493, 2000, 64, 2000, 5, device, "./outputs_2000_2000")

print("Total Time (1 section): " + str(time.time() - seconds) + " seconds")


seconds = time.time()

deconvolve_cloud(7000, 10000, 493, 2000, 64, 1000, 5, device, "./outputs_2000_1000")

print("Total Time (4 sections): " + str(time.time() - seconds) + " seconds")

seconds = time.time()

deconvolve_cloud(7000, 10000, 493, 2000, 64, 500, 5, device, "./outputs_2000_500")

print("Total Time (100 section): " + str(time.time() - seconds) + " seconds")

seconds = time.time()

deconvolve_cloud(7000, 10000, 493, 2000, 64, 100, 5, device, "./outputs_2000_100")

print("Total Time (400 sections): " + str(time.time() - seconds) + " seconds")
from commands import generate_initial_psf, RL_deconv_blind, unroll_psf, clip_psf, intensity_match_image
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

def deconvolve_cloud_section(x, y, z, xy_size, slices, section_size, iterations, device, output_dir):
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

    # blurred = scipy.ndimage.gaussian_filter(img, sigma)

    blurred = torch.clone(img_tensor)


    output_img = np.zeros((img_tensor.shape))

    num = 0

    for i in tqdm(range(int(xy_size / section_size))):
        for j in range(int(xy_size / section_size)):
            blurred_section = torch.clone((blurred[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)]))

            psf_guess = generate_initial_psf(blurred_section)

            output, output_psf = RL_deconv_blind(blurred_section, torch.from_numpy(psf_guess), target_device=device, iterations=iterations)

            output_img[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = output

            if i == 0 and j == 0:
                tiff.imwrite(imgs_dir + "/img_" + str(num) + ".tiff", img_tensor.detach().cpu().numpy())
                tiff.imwrite(output_functions_dir + "/img_" + str(num) + ".tiff", output_psf)

            if i == int(xy_size / section_size) - 1 and j == int(xy_size / section_size) - 1:
                tiff.imwrite(deconv_dir + "/img_" + str(num) + ".tiff", output_img.astype(np.uint16))

            num += 1


def deconvolve_cloud(x, y, z, xy_size, slices, section_size, iterations, device, output_dir):
    vol = CloudVolume('precomputed://https://ntracer2.cai-lab.org/data2/051524_bitbow_ch0', parallel=True, progress=True)

    img = vol[x:x+xy_size, y:y+xy_size, z:z+slices]

    img = img [:, :, :, 0]

    img = img.transpose(2, 0, 1)

    deconv_dir = output_dir + "/deconv"
    output_functions_dir = output_dir + "/functions"
    imgs_dir = output_dir + "/imgs"
    initial_functions_dir = output_dir + "/initial_functions"

    dirs = [deconv_dir, output_functions_dir, imgs_dir, initial_functions_dir]

    for i in dirs:
        clear_dir(i)

    img_tensor = torch.from_numpy(np.array(img).astype(np.int16))
    num = 0
    psf_guess = generate_initial_psf(img_tensor)
    output = torch.clone(img_tensor)

    start_time = time.time()

    iters = []
    times = []



    for i in tqdm(range(int(iterations / 10))):



        output, psf_guess = RL_deconv_blind(output.type(torch.cdouble), torch.from_numpy(psf_guess).type(torch.cdouble), target_device=device, iterations=10)

        output_copy = np.copy(intensity_match_image(img, output))

        output = torch.from_numpy(output)

        iters.append((num + 1) * 10)
        times.append(time.time() - start_time)

        tiff.imwrite(imgs_dir + "/img_" + str((num + 1) * 10) + ".tiff", img_tensor.detach().cpu().numpy())
        tiff.imwrite(output_functions_dir + "/img_" + str((num + 1) * 10) + ".tiff", clip_psf(unroll_psf(psf_guess)))
        tiff.imwrite(deconv_dir + "/img_" + str((num + 1) * 10) + ".tiff", output_copy.astype(np.uint16))

        num +=1

    plt.plot(iters, times)
    plt.xlabel("Iterations")
    plt.ylabel("Time (s)")

    plt.title("Deconvolution Iteration time")

    plt.savefig(output_dir + "/time.png")

    plt.close()


seconds = time.time()

deconvolve_cloud_section(7000, 10000, 493, 1000, 64, 100, 200, device, "/mnt/turbo/jfeggerd/outputs_sectioned")
deconvolve_cloud_section(20000, 6000, 480, 1000, 64, 100, 200, device, "/mnt/turbo/jfeggerd/outputs_2_sectioned")
deconvolve_cloud_section(12000, 8000, 480, 1000, 64, 100, 200, device, "/mnt/turbo/jfeggerd/outputs_3_sectioned")

print("Total Time: " + str(time.time() - seconds) + " seconds")
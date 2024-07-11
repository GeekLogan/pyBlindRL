from commands import generate_initial_psf, RL_deconv_blind, unroll_psf, clip_psf, intensity_match_image, RL_deconv
from utility import clear_dir
import scipy
import skimage.metrics
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

def deconvolve_cloud(x, y, z, xy_size, slices, section_size, iterations, device, output_dir):
    vol = CloudVolume('precomputed://https://ntracer2.cai-lab.org/data2/051524_bitbow_ch0', parallel=True, progress=True)

    img = vol[x:x+xy_size, y:y+xy_size, z:z+slices]

    img = img [:, :, :, 0]

    img = img.transpose(2, 0, 1)

    deconv_dir = output_dir + "/deconv"
    deconv_avg_dir = output_dir + "/deconv_avg"
    output_functions_dir = output_dir + "/functions"
    imgs_dir = output_dir + "/imgs"

    dirs = [deconv_dir, output_functions_dir, imgs_dir, deconv_avg_dir]

    for i in dirs:
        clear_dir(i)

    img_tensor = torch.from_numpy(np.array(img).astype(np.int16))
    num = 0

    psf_guess = np.zeros(np.array(img).shape, dtype=np.complex128)

    psf_piece = generate_initial_psf(np.zeros((slices, section_size, section_size)))

    for i in range(int(xy_size / section_size)):
        for j in range(int(xy_size / section_size)):
            psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = psf_piece

    output = torch.clone(img_tensor)

    start_time = time.time()

    iters = []
    times = []
    ssims = []
    psnrs = []

    ssims_avg = []
    psnrs_avg = []

    for i in tqdm(range(int(iterations / 10))):

        for i in range(int(xy_size / section_size)):
            for j in range(int(xy_size / section_size)):
                output_piece, psf_guess_piece = RL_deconv_blind(img_tensor[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), torch.from_numpy(psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)]).type(torch.cdouble), target_device=device, iterations=10)

                output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = torch.from_numpy(output_piece)
                psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = psf_guess_piece

        ## Data Collection

        iters.append((num + 1) * 10)
        times.append(time.time() - start_time)

        output_copy = np.copy(output.numpy())
        output_copy = intensity_match_image(img_tensor.numpy().astype(np.uint16), output_copy.astype(np.uint16))

        ssims.append(skimage.metrics.structural_similarity(img_tensor.numpy()[32, :, :].astype(np.uint16), output_copy[32, :, :].astype(np.uint16)))
        psnrs.append(skimage.metrics.peak_signal_noise_ratio(img_tensor.numpy()[32, :, :].astype(np.uint16), output_copy[32, :, :].astype(np.uint16)))

        ##

        psf_average = np.zeros((slices, section_size, section_size), dtype=np.complex128)


        #Calculate an average psf from each of the 100 generated
        for i in range(int(xy_size / section_size)):
            for j in range(int(xy_size / section_size)):
                psf_average += psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] / ((xy_size / section_size) * (xy_size / section_size))

        #Apply the average psf to the final image to see how it looks

        average_output = torch.clone(img_tensor)

        for i in range(int(xy_size / section_size)):
            for j in range(int(xy_size / section_size)):

                output_piece= RL_deconv(average_output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), torch.from_numpy(psf_average).type(torch.cdouble), iterations=10)

                average_output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = torch.from_numpy(output_piece)

        ## Data collection for the average

        output_avg_copy = np.copy(average_output.numpy())
        output_avg_copy = intensity_match_image(img_tensor.numpy().astype(np.uint16), output_avg_copy.astype(np.uint16))

        ssims_avg.append(skimage.metrics.structural_similarity(img_tensor.numpy()[32, :, :].astype(np.uint16), output_avg_copy[32, :, :].astype(np.uint16)))
        psnrs_avg.append(skimage.metrics.peak_signal_noise_ratio(img_tensor.numpy()[32, :, :].astype(np.uint16), output_avg_copy[32, :, :].astype(np.uint16)))

        ## End


        tiff.imwrite(imgs_dir + "/img_" + str((num + 1) * 10) + ".tiff", img_tensor.detach().cpu().numpy())
        tiff.imwrite(output_functions_dir + "/img_" + str((num + 1) * 10) + ".tiff", clip_psf(unroll_psf(psf_average)).astype(np.uint16))
        tiff.imwrite(deconv_dir + "/img_" + str((num + 1) * 10) + ".tiff", output.numpy().astype(np.uint16))
        tiff.imwrite(deconv_avg_dir + "/img_" + str((num + 1) * 10) + ".tiff", average_output.numpy().astype(np.uint16))

        num +=1

    plt.plot(iters, times)
    plt.xlabel("Iterations")
    plt.ylabel("Time (s)")

    plt.title("Deconvolution Iteration time")

    plt.savefig(output_dir + "/time.png")

    plt.close()

    plt.plot(iters, ssims)
    plt.xlabel("Iterations")
    plt.ylabel("Structural Similarity")

    plt.title("Structural Similarity Through Iterations")

    plt.savefig(output_dir + "/ssim.png")

    plt.close()

    plt.plot(iters, psnrs)
    plt.xlabel("Iterations")
    plt.ylabel("Peak Signal to Noise Ratio")

    plt.title("PSNR Through Iterations")

    plt.savefig(output_dir + "/psnr.png")

    plt.close()

    plt.plot(iters, ssims_avg)
    plt.xlabel("Iterations")
    plt.ylabel("Structural Similarity Avg PSF")

    plt.title("Structural Similarity Through Iterations")

    plt.savefig(output_dir + "/ssim_avg.png")

    plt.close()

    plt.plot(iters, psnrs_avg)
    plt.xlabel("Iterations")
    plt.ylabel("Peak Signal to Noise Ratio")

    plt.title("PSNR Through Iterations Avg PSF")

    plt.savefig(output_dir + "/psnr_avg.png")

    plt.close()

seconds = time.time()

device = torch.device("cuda", 0)

deconvolve_cloud(7000, 10000, 493, 1000, 64, 100, 300, device, "/mnt/turbo/jfeggerd/outputs_sectioned")
# deconvolve_cloud(20000, 6000, 480, 1000, 64, 100, 100, device, "/mnt/turbo/jfeggerd/outputs_2_sectioned")
# deconvolve_cloud(12000, 8000, 480, 1000, 64, 100, 100, device, "/mnt/turbo/jfeggerd/outputs_3_sectioned")

print("Total Time: " + str(time.time() - seconds) + " seconds")
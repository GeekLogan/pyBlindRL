from commands import generate_initial_psf, RL_deconv_blind, unroll_psf, clip_psf, intensity_match_image, RL_deconv, roll_psf, normalize_psf, edge_correction
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
from skimage.exposure import match_histograms
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
    avg_iters = []
    times = []
    ssims = []
    psnrs = []

    ssims_avg = []
    psnrs_avg = []

    for _ in tqdm(range(int(iterations))):

        for i in range(int(xy_size / section_size)):
            for j in range(int(xy_size / section_size)):
                output_piece, psf_guess_piece = RL_deconv_blind(img_tensor[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), torch.from_numpy(psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)]).type(torch.cdouble), target_device=device, iterations=1, reg_factor=0)

                output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = torch.from_numpy(output_piece)
                psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = psf_guess_piece

        ## Data Collection

        iters.append((num + 1))
        times.append(time.time() - start_time)

        output_copy = np.copy(output.numpy())
        output_copy = intensity_match_image(img_tensor.numpy().astype(np.uint16), output_copy.astype(np.uint16))

        ssims.append(skimage.metrics.structural_similarity(img_tensor.numpy()[32, :, :].astype(np.uint16), output_copy[32, :, :].astype(np.uint16)))
        psnrs.append(skimage.metrics.peak_signal_noise_ratio(img_tensor.numpy()[32, :, :].astype(np.uint16), output_copy[32, :, :].astype(np.uint16)))

        ##



        if (num + 1) % 30 == 0:

            psf_average = np.zeros((slices, section_size, section_size), dtype=np.complex128)
            #Calculate an average psf from each of the 100 generated

            for i in range(int(xy_size / section_size)):
                for j in range(int(xy_size / section_size)):
                    psf_average += normalize_psf(psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] / ((xy_size / section_size) * (xy_size / section_size)))

            #Apply the average psf to the final image to see how it looks

            average_output = torch.clone(img_tensor)

            #Deconvolve with the average psf for about as many iterations as the normal image has had itself

            for it in range(40):
                for i in range(int(xy_size / section_size)):
                    for j in range(int(xy_size / section_size)):
                        output_piece = RL_deconv(img_tensor[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), average_output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), torch.from_numpy(psf_average).type(torch.cdouble), iterations=500, target_device=device)

                        average_output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = torch.from_numpy(output_piece)


                corrected_output = torch.clone(average_output)

                corrected_output = corrected_output.numpy().astype(np.uint16)

                for i in range(int(xy_size / section_size)):
                    for j in range(int(xy_size / section_size)):
                        corrected_output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = match_histograms(corrected_output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)], img_tensor[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].numpy())

                tiff.imwrite(deconv_avg_dir + "/img_" + str((num + 1)) +  "_" + str((it + 1) * 500) + ".tiff", corrected_output.astype(np.uint16))



            ## Data collection for the average
            avg_iters.append(num + 1)
            output_avg_copy = np.copy(average_output.numpy())
            output_avg_copy = intensity_match_image(img_tensor.numpy().astype(np.uint16), output_avg_copy.astype(np.uint16))

            ssims_avg.append(skimage.metrics.structural_similarity(img_tensor.numpy()[32, :, :].astype(np.uint16), output_avg_copy[32, :, :].astype(np.uint16)))
            psnrs_avg.append(skimage.metrics.peak_signal_noise_ratio(img_tensor.numpy()[32, :, :].astype(np.uint16), output_avg_copy[32, :, :].astype(np.uint16)))

            tiff.imwrite(output_functions_dir + "/img_" + str((num + 1)) + ".tiff", clip_psf(unroll_psf(psf_average)).astype(np.uint16))

            ## End

        tiff.imwrite(imgs_dir + "/img_" + str((num + 1)) + ".tiff", img_tensor.detach().cpu().numpy())
        tiff.imwrite(deconv_dir + "/img_" + str((num + 1)) + ".tiff", output.numpy().astype(np.uint16))

        num +=1

def deconvolve_cloud_rolling(x, y, z, xy_size, slices, section_size, iterations, device, output_dir):

    start_time = time.time()

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

    iters = []
    avg_iters = []
    times = []
    ssims = []
    psnrs = []

    ssims_avg = []
    psnrs_avg = []

    setup_time = time.time()
    print("Setup Time:")
    print(setup_time - start_time)

    for i in range(int(xy_size / section_size)):
        for j in range(int(xy_size / section_size)):
            for _ in tqdm(range(int(iterations))):
                start_mem = torch.cuda.memory_allocated(device)
                output_piece, psf_guess_piece = RL_deconv_blind(img_tensor[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), torch.from_numpy(psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)]).type(torch.cdouble), target_device=device, iterations=1, reg_factor=0)


                output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = torch.from_numpy(output_piece)
                psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = psf_guess_piece

    psf_time = time.time()
    print("Blinded Time")
    print(psf_time - setup_time)

    ## Data Collection

    iters.append((num + 1))
    times.append(time.time() - start_time)

    output_copy = np.copy(output.numpy())
    output_copy = intensity_match_image(img_tensor.numpy().astype(np.uint16), output_copy.astype(np.uint16))

    ssims.append(skimage.metrics.structural_similarity(img_tensor.numpy()[32, :, :].astype(np.uint16), output_copy[32, :, :].astype(np.uint16)))
    psnrs.append(skimage.metrics.peak_signal_noise_ratio(img_tensor.numpy()[32, :, :].astype(np.uint16), output_copy[32, :, :].astype(np.uint16)))

    psf_average = np.copy(psf_guess)
    #Calculate an average psf from each of the 100 generated

    psf_average = unroll_psf(psf_average)
    psf_average_small = clip_psf(psf_average, (64, 100, 100))
    psf_average_large = clip_psf(psf_average,  (64, 150, 150))

    psf_average_small = roll_psf(psf_average_small)
    psf_average_large = roll_psf(psf_average_large)

    average_output = np.zeros((12, 12, 64, 150, 150))

    intermediate_output = torch.clone(img_tensor)

    section_size = 100
    overlap = 25

    #Deconvolve with the average psf for about as many iterations as the normal image has had itself

    for i in range(int(xy_size / section_size)):
        for j in range(int(xy_size / section_size)):

            if i > 0 and i < 11 and j > 0 and j < 11:
                output_piece = RL_deconv(img_tensor[:, (i*section_size) - overlap:((i+1) * section_size) + overlap, (j*section_size) - overlap :((j+1) * section_size) + overlap], intermediate_output[:, (i*section_size) - overlap :((i+1) * section_size) + overlap, (j*section_size) - overlap :((j+1) * section_size) + overlap].type(torch.cdouble), torch.from_numpy(psf_average_large).type(torch.cdouble), iterations = iterations, target_device=device)

                # output_piece = match_histograms(output_piece, img_tensor[:, (i*section_size) - overlap:((i+1) * section_size) + overlap, (j*section_size) - overlap :((j+1) * section_size) + overlap].numpy())
                average_output[i, j, :, :, :] = output_piece
            else:
                output_piece = RL_deconv(img_tensor[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)], intermediate_output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), torch.from_numpy(psf_average_small).type(torch.cdouble), iterations = iterations, target_device=device)

    average_output = torch.from_numpy(edge_correction(average_output, 150, 100, (64, 1200, 1200)))

    normal_deconv_time = time.time()
    print("Normal Deconvolution Time")
    print(normal_deconv_time - psf_time)

    ## Data collection for the average
    # avg_iters.append(num + 1)
    # output_avg_copy = np.copy(average_output.numpy())
    # output_avg_copy = intensity_match_image(img_tensor.numpy().astype(np.uint16), output_avg_copy.astype(np.uint16))

    # ssims_avg.append(skimage.metrics.structural_similarity(img_tensor.numpy()[32, :, :].astype(np.uint16), output_avg_copy[32, :, :].astype(np.uint16)))
    # psnrs_avg.append(skimage.metrics.peak_signal_noise_ratio(img_tensor.numpy()[32, :, :].astype(np.uint16), output_avg_copy[32, :, :].astype(np.uint16)))


    tiff.imwrite(deconv_avg_dir + "/img_" + str((num + 1)) + ".tiff", average_output.numpy().astype(np.uint16))
    # tiff.imwrite(output_functions_dir + "/img_" + str((num + 1)) + ".tiff", clip_psf(unroll_psf(psf_average)).astype(np.uint16))

    ## End

    # tiff.imwrite(imgs_dir + "/img_" + str((num + 1)) + ".tiff", img_tensor.detach().cpu().numpy())
    # tiff.imwrite(deconv_dir + "/img_" + str((num + 1)) + ".tiff", output.numpy().astype(np.uint16))


device = torch.device("cuda", 0)
deconvolve_cloud_rolling(6900, 11900, 493, 1200, 64, 1200, 20, device, "/mnt/turbo/jfeggerd/outputs_rolling_edge")
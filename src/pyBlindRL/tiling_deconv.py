from commands import *
from utility import clear_dir
import skimage.metrics
import time
import tifffile as tiff
import torch
from cloudvolume import CloudVolume
from tqdm import tqdm
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


def xy_tiled_image_deconvolution(x, y, z, xy_size, slices, section_size, blind_iterations, normal_iterations, device, output_dir):

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

    psf_guess = np.zeros(np.array(img).shape, dtype=np.complex128)

    psf_piece = generate_initial_psf(np.zeros((slices, section_size, section_size)))

    for i in range(int(xy_size / section_size)):
        for j in range(int(xy_size / section_size)):
            psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = psf_piece

    output = torch.clone(img_tensor)

    setup_time = time.time()
    print("Setup Time:")
    print(setup_time - start_time)

    for i in range(int(xy_size / section_size)):
        for j in range(int(xy_size / section_size)):
            for _ in tqdm(range(int(blind_iterations))):

                output_piece, psf_guess_piece, mem = RL_deconv_blind(img_tensor[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)].type(torch.cdouble), torch.from_numpy(psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)]).type(torch.cdouble), target_device=device, iterations=1, reg_factor=0)

                output[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = torch.from_numpy(output_piece)
                psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)] = psf_guess_piece

    psf_time = time.time()
    print("Blinded Time")
    print(psf_time - setup_time)


    psf_average = np.zeros((slices, section_size, section_size))


    ##Calculate an average psf from those already generated
    psf_average = np.zeros((slices, section_size, section_size), dtype=np.complex128)

    for i in range(int(xy_size / section_size)):
        for j in range(int(xy_size / section_size)):
            psf_average[:, :, :] += psf_guess[:, (i*section_size):((i+1) * section_size), (j*section_size):((j+1) * section_size)]

    psf_average = psf_average / ((xy_size / section_size) **2)

    psf_average = unroll_psf(psf_average)

    tiff.imwrite(output_functions_dir + "/img_" + str((1)) + ".tiff", clip_psf(psf_average).astype(np.uint16))

    psf_average_small = np.copy(psf_average)
    psf_average_large = np.zeros((slices, 150, 150))

    if section_size > 150:
        psf_average_large = clip_psf(psf_average, (64, 150, 150))
        psf_average_small = clip_psf(psf_average, (64, 100, 100))
    else:
        psf_average_large = np.zeros((slices, 150, 150))
        psf_average_large = emplace_center(psf_average_large, psf_average)

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
                output_piece, mem = RL_deconv(img_tensor[:, (i*section_size) - overlap:((i+1) * section_size) + overlap, (j*section_size) - overlap :((j+1) * section_size) + overlap], intermediate_output[:, (i*section_size) - overlap :((i+1) * section_size) + overlap, (j*section_size) - overlap :((j+1) * section_size) + overlap].type(torch.cdouble), torch.from_numpy(psf_average_large).type(torch.cdouble), iterations = normal_iterations, target_device=device)

                average_output[i, j, :, :, :] = output_piece

    average_output = torch.from_numpy(edge_correction(average_output, 150, 100, (64, 1200, 1200)))

    normal_deconv_time = time.time()
    print("Normal Deconvolution Time")
    print(normal_deconv_time - psf_time)

    print("PSNR Ratio Score")
    psnr = skimage.metrics.peak_signal_noise_ratio(img_tensor[:, 100:1100, 100:1100].numpy().astype(np.uint16), intensity_match_image(img_tensor[:, 100:1100, 100:1100].numpy().astype(np.uint16), average_output[:, 100:1100, 100:1100].numpy().astype(np.uint16)))
    print(psnr)

    tiff.imwrite(deconv_avg_dir + "/img_" + str((1)) + ".tiff", average_output.numpy().astype(np.uint16))

device = torch.device("cuda", 0)
# xy_tiled_image_deconvolution(6900, 9900, 493, 1200, 64, 100, 50, 1000, device, "/mnt/turbo/jfeggerd/outputs_rolling_edge")


def z_tiled_image_deconvolution(x, y, z, xy_size, slices, section_size, blind_iterations, normal_iterations, device, output_dir, trial_name):

    start_time = time.time()

    vol = CloudVolume('precomputed://https://ntracer2.cai-lab.org/data2/051524_bitbow_ch0', parallel=True, progress=True)

    img = vol[x:x+xy_size, y:y+xy_size, z:z+slices]

    img = img [:, :, :, 0]

    img = img.transpose(2, 0, 1)

    if os.path.exists(output_dir + "/" + trial_name):
        shutil.rmtree(output_dir + "/" + trial_name)

    os.mkdir(output_dir + "/" + trial_name)

    deconv_dir = output_dir + "/" + trial_name +  "/deconv"
    imgs_dir = output_dir + "/" + trial_name + "/imgs"

    os.mkdir(deconv_dir)
    os.mkdir(imgs_dir)

    img_tensor = torch.from_numpy(np.array(img).astype(np.int16))

    tiff.imwrite(imgs_dir + "/img.tiff", img_tensor.numpy().astype(np.uint16))

    psf_guess = generate_initial_psf(np.zeros((slices, xy_size, xy_size)))

    psf_guess[:, :, :] = psf_guess[32, :, :] #just take the middle layer of the PSF guess match it for all layers

    psf_guess = torch.from_numpy(psf_guess)

    output = torch.clone(img_tensor)

    setup_time = time.time()
    print("Setup Time:")
    print(setup_time - start_time)

    blind_mem = 0

    for i in range(slices):
        for _ in tqdm(range(int(blind_iterations))):
            output_piece, psf_guess_piece, blind_mem = RL_deconv_blind(img_tensor[i, :, :].type(torch.cdouble), output[i, :, :].type(torch.cdouble), psf_guess[i, :, :].type(torch.cdouble), target_device=device, iterations=1, reg_factor=0)
            output[i, :, :] = torch.from_numpy(output_piece)
            psf_guess[i, :, :] = torch.from_numpy(psf_guess_piece)

    psf_time = time.time()
    print("Blinded Time")
    print(psf_time - setup_time)

    output = torch.clone(img_tensor)

    psnr_values = np.zeros((2, int(normal_iterations/10)))

    normal_mem = 0

    for i in tqdm(range(int(slices))):
        for it in range(normal_iterations):
            output_piece, normal_mem = RL_deconv_2D(img_tensor[i, :, :].type(torch.cdouble), output[i, :, :].type(torch.cdouble), psf_guess[i, :, :].type(torch.cdouble), iterations = 1, target_device=device)
            output[i, :, :] = torch.from_numpy(output_piece)
            if (it % 10) == 0:
                psnr = skimage.metrics.peak_signal_noise_ratio(img_tensor[i, :, :].numpy().astype(np.uint16), output[i, :, :].numpy().astype(np.uint16))
                psnr_values[0, int(it / 10)] = it
                psnr_values[1, int(it / 10)] = psnr

    normal_deconv_time = time.time()
    print("Normal Deconvolution Time")
    print(normal_deconv_time - psf_time)

    blended = slice_blending(output.numpy().astype(np.uint16), output.numpy().shape)

    tiff.imwrite(deconv_dir + "/blended" + ".tiff", blended)
    tiff.imwrite(deconv_dir + "/deconv" + ".tiff", output.numpy().astype(np.uint16))

    f = open("data.txt", "w")

    f.write(trial_name)
    f.write("\nTotal runtime: " + str(time.time() - start_time) + "\n")
    f.write("Blind: Its = " + str(blind_iterations) + ", Memory = " + str(blind_mem / 1e6) + " MB\n")
    f.write("Normal: Its = " + str(normal_iterations) + ", Memory = " + str(normal_mem / 1e6) + " MB\n")
    f.close()

    plt.figure()
    plt.plot(psnr_values[0, :], psnr_values[1, :])

    plt.ylabel("PSNR Score")
    plt.xlabel("Normal Deconvolution Iterations")
    plt.savefig(output_dir + "/" + trial_name + "/psnr.png")
    plt.close()


device = torch.device("cuda", 0)

z_tiled_image_deconvolution(7000, 10000, 493, 1000, 64, 64, 25, 500, device, "/mnt/turbo/jfeggerd/outputs_z_tiled", "trial_1")


from commands import generate_initial_psf, RL_deconv_blind
from utility import clear_dir
import cv2
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

vol = CloudVolume('precomputed://https://ntracer2.cai-lab.org/data2/051524_bitbow_ch0', parallel=True, progress=True)

x_lim = 28000
y_lim = 28000
count = 0

imgs = []

dirs = [blurred_dir, deconv_dir, output_functions_dir, imgs_dir, initial_functions_dir]

for i in dirs:
    clear_dir(i)

count = 0

while count < 1:
    z = random.randint(300,1000)
    x = random.randint(1, (x_lim / 2000)) * 2000
    y = random.randint(1, (y_lim / 2000)) * 2000

    img = vol[x-1000:x, y-1000:y, z-64:z]

    img = img [:, :, :, 0]

    img = img.transpose(2, 0, 1)

    if np.std(img) < 17:
        continue

    count += 1

    imgs.append(img)

num = 0

for img in tqdm(imgs):

    img_tensor = torch.from_numpy(np.array(img).astype(np.int16))

    psf_guess = generate_initial_psf(img_tensor)
    output_img, output_psf = RL_deconv_blind(img_tensor, torch.from_numpy(psf_guess), device=device)

    tiff.imwrite(imgs_dir + "/img_" + str(num) + ".tiff", img_tensor.detach().cpu().numpy())
    # tiff.imwrite(blurred_dir + "/img_" + str(num) + ".png", blurred)
    tiff.imwrite(deconv_dir + "/img_" + str(num) + ".tiff", output_img)
    tiff.imwrite(output_functions_dir + "/img_" + str(num) + ".tiff", output_psf)
    tiff.imwrite(initial_functions_dir + "/img_" + str(num) + ".tiff", psf_guess)

    num += 1
from src.pyBlindRL.commands import *
from src.pyBlindRL.utility import clear_dir
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


vol = CloudVolume('precomputed://https://ntracer2.cai-lab.org/data2/051524_bitbow_ch1', parallel=True, progress=True, mip=-1)

img = vol[0:1000, 0:1000, 0:70]

img = img [:, :, :, 0]

img = img.transpose(2, 0, 1)

tiff.imwrite("/mnt/turbo/jfeggerd/outputs_rolling_edge/test_img.tiff", img)
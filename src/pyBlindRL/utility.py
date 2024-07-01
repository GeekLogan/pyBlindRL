from commands import generate_initial_psf, RL_deconv_blind
import cv2
import torch
import os
from cloudvolume import CloudVolume
import random
from tqdm import tqdm
import numpy as np
import glob


def clear_dir(dir):
    files = glob.glob(dir + "/*")
    if not files:
        return
    for f in files:
        os.remove(f)

import torch
from src.pyBlindRL.tiling_deconv import xy_tiled_image_deconvolution, z_tiled_image_deconvolution, h5_input_deconv


device = torch.device("cuda", 0)

xy_tiled_image_deconvolution(6900, 9900, 493, 1200, 64, 100, 25, 50, 1000, device, "/mnt/turbo/jfeggerd/testing_scripts", "testing_variable_size")

z_tiled_image_deconvolution(11000, 12000, 493, 2000, 100, 50, 25, 100, device, "/mnt/turbo/jfeggerd/outputs_z_tiled", "trial_15", False)

h5_input_deconv(40, 1, 100, device, "/mnt/turbo/jfeggerd/outputs_z_tiled", "trial_21", True)


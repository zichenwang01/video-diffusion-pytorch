import os
from os.path import join
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from video_diffusion_pytorch.video_diffusion_pytorch import video_tensor_to_gif

# Path to the .npy folder
npy_folder = '/scratch/jjparkcv_root/jjparkcv1/chloehjh/data-generation/NS/Shu/kf_2d_re1000/'

for seed_folder in Path(npy_folder).glob("seed*"):
    # Print the current seed folder
    print(f"Processing {seed_folder}")
    
    # Initialize the video tensor
    video = torch.empty(0, 128, 128)  
    
    t = 0
    while os.path.exists(join(seed_folder, f'sol_t{t}_step0.npy')):
        step = 0
        while os.path.exists(join(seed_folder, f'sol_t{t}_step{step}.npy')):
            # Load the .npy file
            field = np.load(join(seed_folder, f'sol_t{t}_step{step}.npy'))
            field = torch.tensor(field)

            # Attach npy to the video tensor
            video = torch.cat((video, field.unsqueeze(0)), dim=0)
            
            step += 1
        t += 1
    
    # Check number of frames
    assert video.shape[0] == 320
    
    # Normalize the video tensor to [0, 1]
    video = (video - video.min()) / (video.max() - video.min())
    
    # Save the video tensor as a .gif file
    video = video.unsqueeze(0)
    video_tensor_to_gif(video, f'data/kf_f=320/video_{seed_folder.name}.gif')

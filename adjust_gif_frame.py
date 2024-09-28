import einops
from os.path import join
from pathlib import Path

import torch
import scipy.io
import numpy as np

from video_diffusion_pytorch.video_diffusion_pytorch import video_tensor_to_gif, gif_to_tensor

# Path to the source .gif folder
source_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/ns_small'

# Path to the target .gif folder
target_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/ns_f=3'

idx = 0
for gif_path in Path(source_folder).glob("*.gif"):    
    # Load the .gif file
    gif = gif_to_tensor(gif_path) # (1, 20, 128, 128)
    
    # Adjust frame
    selected_frames = [0, 4, 8, 12, 16]
    gif = gif[:, selected_frames, :, :]
    
    # Save the .gif file
    video_tensor_to_gif(gif, join(target_folder, gif_path.name))
        
    # exit()
import os
from os.path import join
from pathlib import Path

import torch
import scipy.io
import numpy as np

from video_diffusion_pytorch.video_diffusion_pytorch import video_tensor_to_gif, gif_to_tensor

# Path to the source .gif folder
source_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/kf_f=320'

# Path to the target .gif folder
target_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/kf_f=10'
os.makedirs(target_folder, exist_ok=True)

# Number of frames
num_frames = 320

idx = 0
for gif_path in Path(source_folder).glob("*.gif"):    
    # Load the .gif file
    gif = gif_to_tensor(gif_path)
    
    # Selected 10 frames
    selected_frames = [i for i in range(0, num_frames, num_frames // 10)]
    gif = gif[:, selected_frames, :, :]
    
    # Save the .gif file
    video_tensor_to_gif(gif, join(target_folder, gif_path.name))
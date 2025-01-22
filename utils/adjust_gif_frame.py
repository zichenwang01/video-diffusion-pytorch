import os
from os.path import join
from pathlib import Path

from video_diffusion_pytorch.video_diffusion_pytorch import video_tensor_to_gif, gif_to_tensor

# Path to the source .gif folder
source_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/ns_f=20_eval'

# Path to the target .gif folder
target_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/ns5s_f=5_eval'
os.makedirs(target_folder, exist_ok=True)

# Number of frames
num_frames = 20

idx = 0
for gif_path in Path(source_folder).glob("*.gif"):    
    # Load the .gif file
    gif = gif_to_tensor(gif_path)
    
    # Selected 20 frames
    # selected_frames = [i for i in range(0, num_frames, num_frames // 5)]
    selected_frames = [0, 1, 2, 3, 4]
    gif = gif[:, selected_frames, :, :]
    
    # Save the .gif file
    video_tensor_to_gif(gif, join(target_folder, gif_path.name))
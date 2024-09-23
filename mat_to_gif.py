import torch
import scipy.io
import numpy as np

import einops
from pathlib import Path

from video_diffusion_pytorch.video_diffusion_pytorch import video_tensor_to_gif

# Path to the .mat folder
mat_folder = '/nfs/turbo/jjparkcv-turbo-large/chloehjh/data/ns-Re1000-long'

idx = 0
for mat_path in Path(mat_folder).glob("*.mat"):
    # Print the current .mat file 
    print(f"Processing {mat_path}")
    
    # Load the .mat file
    mat = scipy.io.loadmat(mat_path)
    
    # Load the batched videos tensor
    videos = mat['u']
    videos = einops.rearrange(videos, 'b h w f -> b f h w')
    
    # Save each video as a .gif file
    for video in videos:
        video = torch.tensor(video).unsqueeze(0)
        video_tensor_to_gif(video, f'data/ns_Re=1000_f=20/video_{idx}.gif')
        idx += 1
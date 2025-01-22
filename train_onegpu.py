import os 
import sys
from datetime import datetime

from video_diffusion_pytorch import *

# Global variables
results_folder = f'results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
os.makedirs(results_folder, exist_ok=True)

data_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/kf2000_f=20'

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    num_frames = 20,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    data_folder,                    
    train_batch_size = 4,
    train_lr = 1e-4 / 8,
    save_and_sample_every = 5000,
    train_num_steps = 500000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    results_folder = results_folder
)

trainer.train()
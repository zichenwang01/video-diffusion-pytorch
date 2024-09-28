import os 
import sys
from datetime import datetime

from video_diffusion_pytorch import *

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    num_frames = 5,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

results_folder = f'result_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
os.makedirs(results_folder, exist_ok=True)

trainer = Trainer(
    diffusion,
    # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    # '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/ns_small', 
    '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/ns_Re=1000_f=20',                        
    train_batch_size = 2,
    train_lr = 3e-6,
    save_and_sample_every = 1000,
    train_num_steps = 500000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    results_folder = results_folder
)

trainer.train()
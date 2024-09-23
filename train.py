import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    num_frames = 20,
    timesteps = 100,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/ns_small',                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = 2,
    train_lr = 1e-4,
    save_and_sample_every = 1000,
    train_num_steps = 100000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()
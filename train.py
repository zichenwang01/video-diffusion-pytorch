import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Importing custom libraries
from video_diffusion_pytorch import *

# Global variables
results_folder = f'results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'

data_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/kf_f=10'

image_size = 128

num_frames = 10

# Function to set up the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Address for the master node
    os.environ['MASTER_PORT'] = '12355'      # Port for communication (use a unique free port)
    
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Cleanup function for the distributed process group
def cleanup():
    dist.destroy_process_group()

# Function to train with distributed data parallel
def train_ddp(rank, world_size):
    # Set up the distributed environment
    setup(rank, world_size)
    print(f"Rank {rank} started")

    # Create the Unet3D model
    model = Unet3D(
        dim=64,
        dim_mults=(1, 2, 4, 8),
    ).to(rank)  # Move Unet3D to the correct device (rank)

    # Create the diffusion model and move it to the correct device
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        num_frames=num_frames,
        channels=1,      # Number of channels in the input
        timesteps=1000,   # number of steps
        loss_type='l1'    # L1 or L2
    ).to(rank)  # Move diffusion to the correct device

    print(f"Rank {rank} created model")

    # Wrap the entire diffusion model with DistributedDataParallel
    diffusion_ddp = DDP(diffusion, device_ids=[rank])

    print(f"Rank {rank} created DDP")

    # Create results folder (only by rank 0)
    if rank == 0:
        os.makedirs(results_folder, exist_ok=True)

    # Initialize your trainer with the wrapped diffusion model
    trainer = Trainer(
        rank, world_size,
        diffusion_ddp.module,
        data_folder,                  # Path to your data
        train_batch_size=16,           # Per-GPU batch size
        train_lr=1e-4 / 2,
        save_and_sample_every=500,
        train_num_steps=500000,       # Total training steps
        gradient_accumulate_every=2,  # Gradient accumulation steps
        ema_decay=0.995,              # Exponential moving average decay
        amp=True,                     # Turn on mixed precision
        results_folder=results_folder   
    )

    # Start training
    trainer.train()

    # Cleanup after training
    cleanup()

# Save global variables
def save_global_vars():
    global_vars = {
        'results_folder': results_folder,
        'data_folder': data_folder,
        'image_size': image_size,
        'num_frames': num_frames
    }
    torch.save(global_vars, results_folder + 'configs.pt')

# Main function for spawning processes
def main():
    save_global_vars()
    world_size = torch.cuda.device_count()  # Number of GPUs available
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()

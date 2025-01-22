import os
import sys
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Importing custom libraries
from video_diffusion_pytorch.video_diffusion_pytorch import *

# Global variables
results_folder = f'results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
os.makedirs(results_folder, exist_ok=True)

# data_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/kf2000_f=20'
data_folder = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/ns5s_f=5'

checkpoint = None
# checkpoint = torch.load('/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/results/kf2000_f=20/model-15.pt')

image_size = 128

num_frames = 5

# Function to set up the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Address for the master node
    os.environ['MASTER_PORT'] = '12355'      # Port for communication (use a unique free port)
    
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=100000))

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
        # channels=1,      # Number of channels in the input
        timesteps=2000,   # number of steps
        loss_type='l1'    # L1 or L2
    ).to(rank)  # Move diffusion to the correct device

    if checkpoint is not None:
        # Remove 'module.' prefix from state_dict keys if present
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        # Load the model state dict
        diffusion.load_state_dict(new_state_dict)

    # Create results folder (only by rank 0)
    if rank == 0:
        os.makedirs(results_folder, exist_ok=True)

    # Initialize your trainer with the wrapped diffusion model
    trainer = TrainerDDP(
        rank, world_size,
        diffusion,
        data_folder,                  # Path to your data
        train_batch_size=16,           # Per-GPU batch size
        train_lr=1e-5 / 4,
        # train_lr=1e-4 / 4,
        save_and_sample_every=1000,
        train_num_steps=50000,       # Total training steps
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

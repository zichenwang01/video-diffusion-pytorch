import os 
from os.path import join
from pathlib import Path
from datetime import datetime

import torch

from video_diffusion_pytorch.video_diffusion_pytorch import *

# Path to the model checkpoint
model_path = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/results/2024-09-30_22-27-47/model-25.pt'

# Path to the data
data_path = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/kf_f=10/'

# Seed for gt data 
data_seed = 0
# seed = torch.randint(0, 1000, (1,)).item()

# Seed for mask 
mask_seed = 0

# Number of samples to generate
num_samples = 4

# Number of timesteps for diffusion
num_steps = 1000

# Coefficients for the loss functions
obs_coeff, pde_coeff = 1.0, 1.0

def get_ns_loss(u, device=torch.device('cuda')):
    """ Navier-Stokes loss function """
    # Grid parameters
    grid_res = u.size(2)
    grid_step = 1 / (grid_res - 1)
    # Padding
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
    # Laplacian as loss
    loss = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] + 
           u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 
           4 * u[:, :, :, :]) / grid_step**2
    loss = loss.squeeze()
    # Remove boundary loss
    loss[0, :] = 0
    loss[-1, :] = 0
    loss[:, 0] = 0
    loss[:, -1] = 0
    return loss

def random_index(k, grid_size, seed=0, device=torch.device('cuda')):
    """ Randomly mask in k indices from a grid using PyTorch. """
    # Set random seed 
    torch.manual_seed(seed)
    # Generate k unique random indices within the range [0, grid_size^2)
    indices = torch.randperm(grid_size**2, device=device)[:k]
    # Convert flat indices to 2D grid indices (row, column)
    rows, cols = torch.div(indices, grid_size, rounding_mode='floor'), indices % grid_size
    # Create a mask and set selected indices to 1
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)
    mask[rows, cols] = 1
    return mask

def get_mask(shape, num_idx, seed=0):
    """ Get the mask from the index """
    # Get video parameters
    _, frames, grid_size, _ = shape
    # Get grid mask
    mask = random_index(num_idx, grid_size, seed=seed)
    # Expand the mask to the video shape
    mask = mask.unsqueeze(0).unsqueeze(0).expand(1, frames, grid_size, grid_size)
    return mask

def load_model(model_path, device):
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize the model
    unet = Unet3D(
        dim=64,
        dim_mults=(1, 2, 4, 8),
    ).to(device)
    
    # Initialize the Gaussian Diffusion model
    model = GaussianDiffusion(
        denoise_fn=unet,
        image_size=128,  # Example image size, adjust as needed
        num_frames=10,  # Example number of frames, adjust as needed
        channels=3,     # Number of channels in the input
        timesteps=1000,  # Number of timesteps
        loss_type='l1',  # Loss type
        # use_dynamic_thres=False,  # Whether to use dynamic thresholding
        # dynamic_thres_percentile=0.9  # Dynamic threshold percentile
    ).to(device)
    
    # Initialize the EMA model
    ema_model = copy.deepcopy(model)
    
    # Load the model state dict
    model.load_state_dict(checkpoint['model'])
    
    # Load the EMA model state dict
    ema_model.load_state_dict(checkpoint['ema'])
    
    # Initialize the scaler
    scaler = GradScaler('cuda')
    
    # Load the scaler state dict
    scaler.load_state_dict(checkpoint['scaler'])
    
    return model, ema_model, scaler

def dps_samples(
    model, ema_model,
    num_samples, num_steps,
    save_path, device
):
    
    # Load data
    video = gif_to_tensor(data_path + f'video_seed{data_seed}.gif')
    shape = video.shape
    
    # Normalize the data
    video = normalize_img(video)
    
    # Get mask
    mask = get_mask(shape, 100)

    # Get observations
    observations = video * mask
    
    # Observation loss function
    obs_loss_fn = torch.nn.functional.l1_loss
    
    # PDE loss function
    pde_loss_fn = get_ns_loss
    
    # Generate samples using DPS
    videos = ema_model.sample(
        batch_size=num_samples, num_steps=num_steps, 
        observations=observations,
        obs_loss_fn=obs_loss_fn, pde_loss_fn=pde_loss_fn,
        obs_coeff=obs_coeff, pde_coeff=pde_coeff,
        is_dps=True
    )
    print(videos.shape)
    
    # Save the video tensor 
    torch.save(videos, save_path + 'tensor.pt')
    
    # Save the samples as a GIF
    for i in range(num_samples):
        video_tensor_to_gif(videos[i], save_path + f'sample_{i}.gif')

def main():    
    # Path to save the generated samples
    save_path = f'samples/{datetime.now().strftime("%Y-%m-%d")}/'
    os.makedirs(save_path, exist_ok=True)
    
    # Device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model, ema_model, scaler = load_model(model_path, device)
    print("----- Model loaded -----")
    
    # Generate and save samples
    dps_samples(
        model, ema_model,
        num_samples, num_steps,
        save_path, device
    )

if __name__ == '__main__':
    main()
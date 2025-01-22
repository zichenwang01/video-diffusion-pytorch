import os 
import json
from os.path import join
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

from video_diffusion_pytorch.video_diffusion_pytorch import *

# ----------------------------- GLOBAL VARIABLES ----------------------------- 

# Path to the model checkpoint
model_idx = 49
model_path = f'/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/results/ns5s_f=5/model-{model_idx}.pt'

# Path to the data
data_path = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/ns5s_f=5_eval/'

# Seed for gt data 
data_seed = 10001
# seed = torch.randint(0, 1000, (1,)).item()

# Seed for mask 
mask_seed = 0

# Number of samples to generate
num_samples = 1

# Number of timesteps for diffusion
num_steps = 1000

# Number of frames in the video
num_frame = 5

# Number of observations
# num_obs = 800 # 5%
# num_obs = 1600 # 10%
num_obs = 2400 # 15%
# num_obs = 3200 # 20%
# num_obs = 5000 # 30% 

# Coefficient for the observation loss
obs_coeff = 0.01

# Coefficient and starting point for the PDE loss
pde_start = 1.5
pde_coeff = 1

# Path to save samples
# exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_name = f'2_coeff/linear_non-scaled_obs_coeff={obs_coeff}'
save_path = f'samples/{exp_name}/'
os.makedirs(save_path, exist_ok=True)
print(f"----- Saving samples to {save_path} -----")

# ------------------------------- DPS FUNCTIONS -------------------------------

# def obs_loss(u, mask, observations, method='l1'):
#     """ Observation loss function """
#     if method == 'l1':
#         return torch.mean(torch.abs(u * mask - observations))
#     elif method == 'l2':
#         return torch.mean((u * mask - observations)**2)

def obs_loss(u, mask, observations):
    """ Observation loss function """
    residual = u * mask - observations
    obs_step = obs_coeff / torch.norm(residual, 2)
    obs_loss = torch.norm(residual, 2) ** 2
    return obs_step, obs_loss

def ns_loss(u):
    """ Navier-Stokes loss function """
    # Grid parameters
    grid_res = u.size(2)
    grid_step = 1 / (grid_res - 1)
    # Padding
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
    # Laplacian as loss
    loss = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] + 
           u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 
           4 * u_padded[:, :, 1:-1, 1:-1]) / grid_step**2
    loss = loss.squeeze()
    # Remove boundary loss
    loss[0, :] = 0
    loss[-1, :] = 0
    loss[:, 0] = 0
    loss[:, -1] = 0
    # Return the mean squared loss
    return torch.mean(loss**2)

def eval_loss(u, gt):
    """ Relative squared error of the full denoised video """
    u = u.cpu().numpy()
    gt = gt.cpu().numpy()
    rse = np.sum((gt - u) ** 2) / np.sum(gt - np.mean(gt) ** 2) 
    return rse

def random_index(k, grid_size, seed, device):
    """ Randomly mask in k indices from a grid using PyTorch. """
    # Set random seed 
    torch.manual_seed(seed)
    # Generate k unique random indices within the range [0, grid_size^2)
    indices = torch.randperm(grid_size**2)[:k]
    # Convert flat indices to 2D grid indices (row, column)
    rows, cols = torch.div(indices, grid_size, rounding_mode='floor'), indices % grid_size
    # Create a mask and set selected indices to 1
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    mask[rows, cols] = 1
    return mask

def get_consistent_mask(shape, num_idx, seed, device):
    """ Get the mask from the index """
    # Get video parameters
    channels, frames, grid_size, _ = shape
    # Get grid mask
    mask = random_index(num_idx, grid_size, seed=seed, device=device)
    # Expand the mask to the video shape
    mask = mask.unsqueeze(0).unsqueeze(0).expand(channels, frames, grid_size, grid_size)
    return mask

def get_init_mask(shape, num_idx, seed, device):
    """ Get the mask from the index """
    # Get video parameters
    _, frames, grid_size, _ = shape
    # Get grid mask
    init_mask = random_index(num_idx, grid_size, seed=seed, device=device)
    # Expand the mask to the video shape
    mask = torch.zeros((1, frames, grid_size, grid_size), dtype=torch.float32, device=device)
    mask[:, 0, :, :] = init_mask
    return mask

def get_mask(
    shape, num_obs, 
    seed=0, device=torch.device('cuda'),
    method='consistent'
):
    if method == 'consistent':
        return get_consistent_mask(shape, num_obs, seed, device)
    elif method == 'init':
        return get_init_mask(shape, num_obs, seed, device)

# ------------------------------------ MAIN ------------------------------------

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
        num_frames=num_frame,  # Example number of frames, adjust as needed
        channels=3,     # Number of channels in the input
        timesteps=1000,  # Number of timesteps
        loss_type='l1',  # Loss type
        # use_dynamic_thres=False,  # Whether to use dynamic thresholding
        # dynamic_thres_percentile=0.9  # Dynamic threshold percentile
    ).to(device)
    
    # Initialize the EMA model
    ema_model = copy.deepcopy(model)
    
    # Remove 'module.' prefix from state_dict keys if present
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Load the model state dict
    model.load_state_dict(new_state_dict)
    
    # Load the EMA model state dict
    ema_state_dict = checkpoint['ema']
    new_ema_state_dict = {}
    for k, v in ema_state_dict.items():
        if k.startswith('module.'):
            new_ema_state_dict[k[7:]] = v
        else:
            new_ema_state_dict[k] = v
    
    # Load the EMA model state dict
    ema_model.load_state_dict(new_ema_state_dict)
    
    # Initialize the scaler
    scaler = GradScaler('cuda')
    
    # Load the scaler state dict
    scaler.load_state_dict(checkpoint['scaler'])
    
    # Change ema_model timestep
    if ema_model.num_timesteps != num_steps:
        ema_model.__init__(
            denoise_fn=ema_model.denoise_fn,
            image_size=ema_model.image_size,
            num_frames=ema_model.num_frames,
            channels=ema_model.channels,
            timesteps=num_steps,
            loss_type=ema_model.loss_type,
            use_dynamic_thres=ema_model.use_dynamic_thres,
            dynamic_thres_percentile=ema_model.dynamic_thres_percentile
        )
        ema_model = ema_model.to(device)
    
    return model, ema_model, scaler

def dps_samples(
    model, ema_model,
    num_samples, num_steps,
    save_path, device
):
    
    # Load gt video
    gt_video = gif_to_tensor(data_path + f'video_{data_seed}.gif')
    # gt_video = gif_to_tensor(data_path + f'video_seed{data_seed}.gif')
    gt_video = gt_video.to(device)
    shape = gt_video.shape
    # print(shape)
    
    # Normalize the gt video
    gt_video = normalize_img(gt_video)
    
    # Get mask
    mask = get_mask(
        shape=shape, num_obs=num_obs, 
        seed=mask_seed, device=device,
        method='consistent'
    )

    # Get observations
    observations = gt_video * mask
    
    # Generate samples using DPS
    videos = ema_model.sample(
        batch_size=num_samples, num_steps=num_steps, 
        is_dps=True, mask=mask, observations=observations, 
        obs_loss_fn=obs_loss, pde_loss_fn=ns_loss,
        pde_start_ratio=pde_start,
        obs_coeff=obs_coeff, pde_coeff=pde_coeff,
    )
    
    # Save config 
    gt_video = unnormalize_img(gt_video) 
    config = {
        'model_idx': model_idx,
        'data_seed': data_seed,
        'mask_seed': mask_seed,
        'num_samples': num_samples,
        'num_steps': num_steps,
        'num_obs': num_obs,
        'obs_coeff': obs_coeff,
        'pde_coeff': pde_coeff,
        'pde_start': pde_start,
        'loss': float(eval_loss(videos[0], gt_video))
    }
    with open(save_path + 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save tensors
    torch.save(gt_video, save_path + f'gt_seed{data_seed}.pt') # gt
    torch.save(mask, save_path + 'mask.pt') # mask
    observations = unnormalize_img(observations)
    torch.save(observations, save_path + 'obs.pt') # obs
    torch.save(videos, save_path + 'dps.pt') # dps
    
    # Save images
    os.makedirs(save_path + 'gt_images/', exist_ok=True) # gt
    for f in range(shape[1]):
        tensor_to_image(gt_video[:, f, :, :], save_path + f'gt_images/frame{f}.png')
    os.makedirs(save_path + 'mask_images/', exist_ok=True) # mask
    for f in range(shape[1]):
        tensor_to_image(mask[:, f, :, :], save_path + f'mask_images/frame{f}.png')
    os.makedirs(save_path + 'obs_images/', exist_ok=True) # obs
    for f in range(shape[1]):
        tensor_to_image(observations[:, f, :, :], save_path + f'obs_images/frame{f}.png')
    os.makedirs(save_path + 'dps_images/', exist_ok=True) # dps
    for i in range(num_samples):
        os.makedirs(save_path + f'dps_images/sample_{i}/', exist_ok=True)
        for f in range(shape[1]):
            tensor_to_image(videos[i][:, f, :, :], save_path + f'dps_images/sample_{i}/frame{f}.png')
    
    # Save videos
    video_tensor_to_gif(gt_video, save_path + f'gt_seed{data_seed}.gif') # gt
    video_tensor_to_gif(mask, save_path + 'mask.gif') # mask
    video_tensor_to_gif(observations, save_path + 'obs.gif') # obs
    for i in range(num_samples): # dps
        video_tensor_to_gif(videos[i], save_path + f'dps_sample_{i}.gif')

def main():        
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
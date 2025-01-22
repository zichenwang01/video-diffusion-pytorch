import os 
from os.path import join
from pathlib import Path
from datetime import datetime

import torch

from video_diffusion_pytorch.video_diffusion_pytorch import *

# Number of samples to generate
num_samples = 1

# Number of timesteps for diffusion
num_steps = 1000

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
        num_frames=20,  # Example number of frames, adjust as needed
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
    
    return model, ema_model, scaler

def generate_samples(
    model, ema_model,
    num_samples, num_steps,
    save_path, device
):
    # Generate samples
    videos = ema_model.sample(batch_size=num_samples, num_steps=num_steps)
    print(videos.shape)
    
    # Save the video tensor 
    torch.save(videos, save_path + 'tensor.pt')
    
    # Save the samples as a GIF
    for i in range(num_samples):
        video_tensor_to_gif(videos[i], save_path + f'sample_{i}.gif')

def main():
    # Path to the model checkpoint
    model_idx = '15_1'
    model_path = f'/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/results/kf2000_f=20/model-{model_idx}.pt'
    # model_path = f'/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/results/kf_f=20/model-{model_idx}.pt'
    
    # Path to save the generated samples
    # save_path = f'samples/{datetime.now().strftime("%Y-%m-%d")}/'
    save_path = f'samples/kf2000_f=20_model-{model_idx}/'
    os.makedirs(save_path, exist_ok=True)
    print(f"----- Saving samples to {save_path} -----")
    
    # Device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"----- Running on {device} -----")
    
    # Load the model
    model, ema_model, scaler = load_model(model_path, device)
    print("----- Model loaded -----")
    
    # Generate and save samples
    generate_samples(
        model, ema_model,
        num_samples, num_steps,
        save_path, device
    )

if __name__ == '__main__':
    main()
import os 
import json
from os.path import join
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

from video_diffusion_pytorch.video_diffusion_pytorch import *
from sample_dps import *
import sample_dps

# ----------------------------- GLOBAL VARIABLES ----------------------------- 

# Path to the model checkpoint
model_path = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/results/kf_f=20/model-39.pt'

# Path to the data
data_path = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/data/kf_f=20_eval/'

# Seed for gt data 
data_seed = 1111
# seed = torch.randint(0, 1000, (1,)).item()

# Seed for mask 
mask_seed = 0

# Number of samples to generate
num_samples = 1

# Number of timesteps for diffusion
num_steps = 1000

# Number of frames in the video
num_frame = 20

# Number of observations to test
num_obs_values = [800, 1600, 2400, 3200, 5000]

# Observation coefficients to test
obs_coeff_values = [1, 10, 100.0, 1000.0, 10000.0]

# Coefficient for the observation loss
pde_coeff_values = [1e-2, 1e-1, 1]

# Coefficient and starting point for the PDE loss
pde_start_values = [0.0, 0.2, 0.4, 0.6, 0.8]

# Logs of loss values
loss_logs = {}


def main():        
    # Device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model, ema_model, scaler = load_model(model_path, device)
    print("----- Model loaded -----")
    
    # Hyperparameter search
    for num_obs in num_obs_values:
        for obs_coeff in obs_coeff_values:
            for pde_start in pde_start_values:
                for pde_coeff in pde_coeff_values:
                    # Create a unique save path for each combination
                    current_save_path = f'samples/kf_f=20_hparam/num_obs_{num_obs}_obs_coeff_{obs_coeff}_pde_start_{pde_start}_pde_coeff_{pde_coeff}/'
                    os.makedirs(current_save_path, exist_ok=True)
                    print(f"----- Saving samples to {current_save_path} -----")
                    
                    # Synchronize the parameters
                    sample_dps.num_obs = num_obs
                    sample_dps.obs_coeff = obs_coeff
                    sample_dps.pde_coeff = pde_coeff
                    sample_dps.pde_start = pde_start
                    
                    # Generate and save samples
                    dps_samples(
                        model, ema_model,
                        num_samples, num_steps,
                        current_save_path, device
                    )
                    
    # Save loss logs
    with open('samples/kf_f=20_hparam/losses.json', 'w') as f:
        json.dump(loss_logs, f, indent=4)

if __name__ == '__main__':
    main()
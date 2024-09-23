import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion
from video_diffusion_pytorch.video_diffusion_pytorch import video_tensor_to_gif

# ------------------------------- PARAMETERS -------------------------------
num_frames = 5

timesteps = 1000

# --------------------------------- MODEL ---------------------------------
model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    num_frames = num_frames,
    timesteps = timesteps, 
    loss_type = 'l1'    # L1 or L2
)

# videos = torch.randn(1, 3, 5, 32, 32) # video (batch, channels, frames, height, width) - normalized from -1 to +1
# loss = diffusion(videos)
# loss.backward()
# # after a lot of training

# ------------------------------- SAMPLE VIDEO ------------------------------
# sample video
sampled_videos = diffusion.sample(batch_size = 2)
print(sampled_videos.shape) # (batch, channel, frame, res, res)

# prepare video
video = sampled_videos[0]

# save video 
video_tensor_to_gif(video, f'samples/sample_n={num_frames}_t={timesteps}.gif')

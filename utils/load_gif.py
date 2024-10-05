from video_diffusion_pytorch.video_diffusion_pytorch import video_tensor_to_gif, gif_to_tensor

# Path to the .gif file
gif_file_path = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/samples/sampled.gif'

# Load the .gif file
video = gif_to_tensor(gif_file_path)

print(video.shape) # (3, 5, 32, 32)
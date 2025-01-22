import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

gt_path = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/samples/obs=5000_obsw=10000/gt_images'

dps_path = '/nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch/samples/obs=5000_obsw=10000/dps_images/sample_0'

frame = 0

gt = Image.open(gt_path + f'/frame{frame}.png')
gt = np.array(gt)

dps = Image.open(dps_path + f'/frame{frame}.png')
dps = np.array(dps)

relative_error = np.linalg.norm(gt - dps) / np.linalg.norm(gt)

print(f'Relative error: {relative_error}')
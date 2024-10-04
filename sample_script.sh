#!/bin/bash
#SBATCH -J train_video_diffusion		# Job name
#SBATCH -o training.out	                # output file
#SBATCH -e training.err      	        # error log file
#SBATCH --mail-type=ALL	                # request status by email 
#SBATCH --mail-user=zzzichen@umich.edu
#SBATCH -N 1			                # request number of nodes
#SBATCH -n 1			                # request number of cores
#SBATCH --get-user-env	                # retrieve the login environment
#SBATCH --mem=20G	                    # request memory per node
#SBATCH -t 2:00:00		                # request time
#SBATCH --partition=spgpu,spgpu2	    # request partition
#SBATCH --gres=gpu:1	                # request GPU

cd /nfs/turbo/jjparkcv-turbo-large/zichen/video-diffusion-pytorch
python3 sample.py			            # run python file

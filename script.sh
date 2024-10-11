#!/bin/bash

#SBATCH --job-name=colorize        
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=60G                   # Total RAM to be used
#SBATCH --gres=gpu:2             # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p                      # Use the gpu partition
#SBATCH -q cscc-gpu-qos                    # To enable the use of up to 8 GPUs

python notebooks/colorize/colorize.py
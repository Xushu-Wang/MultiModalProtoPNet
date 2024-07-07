#!/usr/bin/env bash

#SBATCH --job-name=resnet_test      # Job name
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb                  # Job memory request
#SBATCH --time=15:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:2
#SBATCH --output=logs/goofy_ahh_resnet_%j.out

eval "$(conda shell.bash hook)" 
conda activate intnn
python3 train_resnet_backbone.py --configs configs/resnet.yaml

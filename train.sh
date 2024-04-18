#!/usr/bin/env bash

#SBATCH --job-name=protopnet_bioscan      # Job name
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                  # Job memory request
#SBATCH --time=15:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:4

eval "$(conda shell.bash hook)" 
conda activate protopnet
python3 main.py --configs="configs/cub.yaml"

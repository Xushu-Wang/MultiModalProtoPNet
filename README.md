# Multi-Modal ProtoPNet

This repo contains a PyTorch implementation for the multimodal version of ProtoPnet. There are three datasets we work with. 
1. The CUB dataset is simply referred to as cub. 
2. The genetics portion of the BIOSCAN dataset is referred to as genetics. 
3. The images portion of the BIOSCAN dataset is referred to as bioscan. 

## Dependencies

First, create an environment and install the necessary dependencies. 
```
conda create -n protopnet python=3.10 
conda activate protopnet 
pip install -r requirements.txt
```

## Dataset Directory Structure 
You want to make datasets folder that contains the relevant data for CUB and Bioscan. To do this, replace the symlinks in `datasets/` as follows 
``` 
dataset/
    bioscan -> /usr/project/xtmp/xw214/bioscan/
    cub200_cropped -> /usr/project/xtmp/xw214/datasets/cub200_cropped/
    genetics -> /usr/project/xtmp/mb625/data/BIOSCAN-1M/
```
You should be able to do this in the Duke cluster since the folders in `xtmp` give global read-only access to all users. All the datasets are augmented and cropped, so no further preprocessing is needed. The tree structure should look something like this. 
```
(base) ~/MultiModalPPNet/datasets/bioscan>tree -L 1
.
├── test
├── test_diptera
├── train
├── train_augmented
├── train_diptera
└── train_diptera_augmented

(base) ~/MultiModalPPNet/datasets/cub200_cropped>tree -L 1 
.
├── test_cropped
├── train_cropped
└── train_cropped_augmented

(base) ~/MultiModalPPNet/datasets/genetics>tree -L 2
.
├── images
│   └── cropped_256
├── large_diptera_family-test.tsv
├── large_diptera_family-train.tsv
├── large_diptera_family-validation.tsv
├── large_insect_order-test.tsv
├── large_insect_order-train.tsv
├── large_insect_order-validation.tsv
├── medium_diptera_family-test.tsv
├── medium_diptera_family-train.tsv
├── medium_diptera_family-validation.tsv
├── metadata_cleaned_columns.txt
├── metadata_cleaned_permissive.tsv
├── metadata_cleaned_restrictive.tsv
├── small_diptera_family-test.tsv
├── small_diptera_family-train.tsv
└── small_diptera_family-validation.tsv
```

## Pretrained Backbone Layers 
We provide pretrained backbone layers (resnet, vgg, and lightweight CNNs) as the feature extractors for ProtoPnet. 
1. To retrieve the pretrained backbone network for the genetics dataset. Download from Charlie's google drive link with `gdown`. 
```
gdown 1qTRhdvujg4FyNNa3s0W-w-XOLkgLcNTW
```
2. For the cub dataset, we use a standard resnet 50. 
3. For the bioscan dataset, we use a standard resnet 50 and vgg 19, but we will train a feature extractor with a custom CNN. 
Make sure to move all the `.pth` files into the `pretrained_backbones/` directory. 

## Running 
To train the models, run the script, where the argument is the `yaml` configuration file. 
``` 
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
python3 main.py --configs="configs/bioscan.yaml"
```

## Running the Experiment (Deprecated)

1. For images, augment the original dataset using augmentation/img_aug.py. The default target directory is root_dir + `train_cropped_augmented/`. (Not completely implemented yet). 

```python img_aug.py --root_dir```

2. In configs/cfg.py, provide the appropriate strings for data_path, train_dir, test_dir, train_push_dir in update_cfg(cfg, args) function:

    - data_path is where the dataset resides
    - train_dir is the directory containing the augmented training set
    - test_dir is the directory containing the test set
    - train_push_dir is the directory containing the original (unaugmented) training set

3. run the main script

```python main.py --dataset [dataset name] --backbone [backbone of protopnet]```

Here ```dataset``` name is one of the following:

- ```bioscan```: This corresponds to the images in The BIOSCAN-1M Insect Dataset.
- ```cub```: This corresponds to The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset.
- ```genetics```: This corresponds to the genetics in The BIOSCAN-1M Insect Dataset.
- ```multimodal```: This corresponds to the full version (images + genetics) of The BIOSCAN-1M Insect Dataset.

Recommended hardware: 4 NVIDIA Tesla P-100 GPUs or 8 NVIDIA Tesla K-80 GPUs


## Dataset Source

- The BIOSCAN-1M Insect Dataset (Original): https://github.com/zahrag/BIOSCAN-1M
- The BIOSCAN-1M Insect Dataset (Preprocessed): https://drive.google.com/drive/folders/1GozrkRDfLurtQ1tMtQ2uq-I8l303PNHW?usp=share_link
- The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset: https://www.vision.caltech.edu/datasets/cub_200_2011/
- The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset (Cropped)

# Multi-Modal ProtoPNet

This repo contains a PyTorch implementation for the multimodal version of protopnet


## Dependencies

First, create an environment and install the necessary dependencies. 
```
conda create -n protopnet python=3.10 
conda activate protopnet 
pip install -r requirements.txt
```

## Dataset Directory Structure 
You want two folders in the base, `datasets` and `bioscan`. Once they are augmented and cropped, you should have a tree structure as
such. 
``` 
.
└── cub200_cropped
    ├── test_cropped
        ├── 001
        ├── 002 
        └── ... 
    ├── train_cropped
    └── train_cropped_augmented
└── bioscan 
    ├── test
    ├── test_diptera
    ├── train
    ├── train_augmented
    ├── train_diptera
    └── train_diptera_augmented
```
If you're in the duke cs cluster, you can create a symlink to Andy's dataset. 
```
ln -s /usr/xtmp/xw214/datasets datasets 
ln -s /usr/xtmp/xw214/bioscan bioscan 
```

## Running the Experiment

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
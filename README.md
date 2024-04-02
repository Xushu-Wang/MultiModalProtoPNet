# Multi-Modal ProtoPNet

This repo contains a PyTorch implementation for the multimodal version of protopnet


## Dependency

The following are packages needed for running this repo.


- PyTorch==2.0.1
- Augmentor==0.2.12
- numpy
- pandas
- cv2
- matplotlib
- yacs

## Running the Experiment

1. For images, augment the original dataset using augmentation/img_aug.py. The default target directory is root_dir + 'train_cropped_augmented/'

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
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

```python main.py --dataset [dataset name] --backbone [backbone of protopnet]```

Here ```dataset``` name is one of the following:

- ```bioscan```: This corresponds to the images in The BIOSCAN-1M Insect Dataset.
- ```cub```: This corresponds to The Caltech-UCSD Birds-200-2011 (CUB-200-2011).
- ```genetics```: This corresponds to the genetics in The BIOSCAN-1M Insect Dataset.
- ```multimodal```: This corresponds to the full version (images + genetics) of The BIOSCAN-1M Insect Dataset.

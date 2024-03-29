import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

from genetics import GeneticCGR, GeneticKmerFrequency, GeneticOneHot


class Bio1MScan(Dataset):

    """
        A dataset class for the BIOSCAN data. Samples are images and unpadded strings of nucleotides, including base pairs A, C, G, T and an unknown character N.

        Args:
            datapath (str): The path to the dataset file (csv or tsv).
            imgpath (str): The path to the dataset file 
            img_transformation (callable, optional): Optional transforms to be applied to the image. Default is None.
            genetic_transformation (callable, optional): Optional transforms to be applied to the genetics. Default is One Hot Encoding.
            genetic level (str): If supplied, the dataset will drop all rows where the given taxonomy level is not present. Default is None.
            genetic_classes (dict): If supplied, the dataset will only include rows where the given taxonomy level is within the given list of classes. Default is None. Use for validation and test sets.
            
        Returns:
            (image, genetic, label)
    """

    def __init__(self,
                 datapath: str,
                 imgpath: str,
                 img_transformation: transforms,
                 genetic_transformation: str = 'onehot',
                 genetic_level: str = None,
                 genetic_classes: dict = None):

        # Read the Genetic and Image Information

        self.genetic_dataset = pd.read_csv(datapath)

        self.img_dataset = datasets.ImageFolder(
            datapath,
            img_transformation)

        # Initialize Genetic Transformation

        if genetic_transformation == 'onehot':
            self.genetic_transform = GeneticOneHot(
                length=720, zero_encode_unknown=True, include_height_channel=True)
        elif genetic_transformation == 'kmer':
            self.genetic_transform = GeneticOneHot()
        elif genetic_transformation == 'cgr':
            self.genetic_transform = GeneticCGR()
        else:
            self.genetic_transform = None

    def __getitem__(self, index):

        path, _ = self.img_dataset.imgs[index]

        img, label = self.img_dataset[index]

        sampleid = os.path.basename(path)

        genetic = self.genetic_dataset[self.genetic_dataset['sampleid']
                                        == sampleid]['nucraw'].values[0]
        
        genetic = self.genetic_transform(genetic)
        
        return img, genetic, label

    def __len__(self):
        return len(self.genetic_dataset)

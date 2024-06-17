import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import regex as re

from dataio.genetics import GeneticCGR, GeneticKmerFrequency, GeneticOneHot


def extract_id(path):
    pattern = r'(?<=_)(?<id>(BIOUG)?\d+(\-[^\.]\d+)?)(?=\.jpg)'
    return re.search(pattern, path).group('id')

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
                 level: str = None,
                 classes: dict = None,
                 sep: str = "\t"):

        # Read the Genetic and Image Information
        self.genetic_data = pd.read_csv(datapath, sep=sep)
        self.level = level
        self.img_dataset = datasets.ImageFolder(
            imgpath,
            img_transformation)

        # Set the index of the genetic dataset to be the image file path
        self.genetic_data = self.genetic_data.set_index("image_path") 

        # Initialize Genetic Transformation
        if genetic_transformation == 'onehot':
            self.genetic_transform = GeneticOneHot(
                length=720, zero_encode_unknown=True, include_height_channel=True)
        elif genetic_transformation == 'kmer':
            self.genetic_transform = GeneticKmerFrequency()
        elif genetic_transformation == 'cgr':
            self.genetic_transform = GeneticCGR()
        else:
            self.genetic_transform = None

        self.taxnomy_level = ["phylum", "class", "order", "family", "subfamily", "tribe", "genus", "species", "subspecies"]
        if self.level:
            if not self.level in self.taxnomy_level:
                raise ValueError(f"drop_level must be one of {self.taxnomy_level}")
            self.genetic_data = self.genetic_data[self.genetic_data[self.level] != "not_classified"]

    def get_classes(self):
        raise NotImplementedError("Note: The order of classes for training and validation come from the same source, the order of filenames in the data folder.")

    def __getitem__(self, index):
        path, _ = self.img_dataset.imgs[index]
        img, label = self.img_dataset[index]
        # print(path)

        file = os.path.split(path)[-1]

        try:
            sample = self.genetic_data.loc[file]

        except KeyError as e:
            print(f"{e} not found in genetic dataset. Are you using a data file generated for this image dataset?")
            exit()
            # print(f"{sampleid} found")
            # print(genetic)
            # print(self.genetic_data[self.genetic_data['sampleid']
            #                             == sampleid]['nucraw'].values)

        genetic = self.genetic_transform(sample["nucraw"])
        
        return img, genetic, label

    def __len__(self):
        return len(self.img_dataset)
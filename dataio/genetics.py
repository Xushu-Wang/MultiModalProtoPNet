import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Any
import torch.nn.functional as F
import numpy as np



class GeneticOneHot(object):
    
    """Map a genetic string to a one-hot encoded tensor, values being in the color channel dimension.

    Args:
        length (int): The length of the one-hot encoded tensor. Samples will be padded (with Ns) or truncated to this length.
        zero_encode_unknown (bool, optional): Whether to encode unknown characters as all zeroes. Otherwise, encode them as (1,0,0,0,0). Default is True.
        include_height_channel (bool, optional): Whether to include a height channel in the one-hot encoding. Default is False.
    """

    def __init__(self, length:int=720, zero_encode_unknown: bool=True, include_height_channel: bool=False):
        self.zero_encode_unknown = zero_encode_unknown
        self.length = length
        self.include_height_channel = include_height_channel

    def __call__(self, genetic_string: str):
        
        """
        Args:
            genetics (str): The genetic data to be transformed.

        Returns:
            torch.Tensor: A one-hot encoded tensor of the genetic data.
        """
        # Create a dictionary mapping nucleotides to their one-hot encoding
        nucleotides = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}

        # Convert string to (1, 2, 1, 4, 0, ...)
        category_tensor = torch.tensor([nucleotides[n] for n in genetic_string])
        
        # Pad and crop
        category_tensor = category_tensor[:self.length]
        category_tensor = F.pad(category_tensor, (0, self.length - len(category_tensor)), value=0)

        # One-hot encode
        onehot_tensor = F.one_hot(category_tensor, num_classes=5).permute(1, 0)
        
        # Drop the 0th channel, changing N (which is [1,0,0,0,0]) to [0,0,0,0] and making only 4 classes
        if self.zero_encode_unknown:
            onehot_tensor = onehot_tensor[1:, :]

        if self.include_height_channel:
            onehot_tensor = onehot_tensor.unsqueeze(1)

        return onehot_tensor.float()


class GeneticKmerFrequency(object):
    """Map a genetic string to a k-mer frequency tensor, with values in the color channel dimensions.
    
    Args:
        length (int): The length of the frequency encoded tensor. Samples will be padded (with 0s) or truncated to this length.
        k (int): The size of each k-mer
    """
    
    def __init__(self, length:int=720, k=4):
        self.length = length
        self.k = k
        self.dna_one_hot_mapping = dict(zip("ACGT", range(4))) 
        
    def one_hot_encode(self, dna_seq):
        
        one_hot = [self.dna_one_hot_mapping[a] for a in dna_seq]
        return np.eye(4)[one_hot].transpose()
    
    
    def k_mer_freq(self, dna_seq, k=4):
    
        one_hot = self.one_hot_encode(dna_seq) # 4 by n
        
        freq = np.sum(one_hot[:, :k], axis=1)
        encoding = [freq.copy()]
        
        for i in range(k, len(one_hot[0])):
            freq += one_hot[:, i] - one_hot[:, i-k]
            encoding.append(freq.copy())
        
        return np.transpose(encoding)

        
    def __call__(self, genetic_string:str):
        """
        Args:
            genetic_string (str): The genetic sequence to be transformed.
            
        Returns:
            torch.Tensor: A 4-by-length tensor encoding frequency
        """
        
        # Create a dictionary mapping nucleotides to their one-hot encoding
        nucleotide_index = {"A": 0, "C": 1, "G": 2, "T": 3}
        
        # Convert string to (1, 2, 1, 4, 0, ...)
        category_tensor = torch.tensor([nucleotide_index[n] for n in genetic_string])
        
        # Pad and crop
        category_tensor = category_tensor[:self.length]
        category_tensor = F.pad(category_tensor, (0, self.length - len(category_tensor)), value=0)

        # One-hot encode
        one_hot = F.one_hot(category_tensor, num_classes=4)
        
        # Sliding window sum k-mer frequencies
        freq = torch.sum(one_hot[:, :self.k], dim=0)
        freq_encoding = [list(freq).copy()]
        
        for i in range(self.k, len(one_hot[0])):
            freq += one_hot[:, i] - one_hot[:, i-self.k]
            freq_encoding.append(list(freq).copy())
            
        return torch.tensor(freq_encoding)
    
    
class GeneticCGR(object):
    
    """Map a genetic string to a chaos game representation, specifically a list of points in a [0,1]x[0,1] region

    Args:
        length (int): The desired length of the encoded sequence. Samples will be padded (with Ns) or truncated to this length.
    """
    
    def __init__(self, length:int=720):
        self.length = length
        
        self.cgt_mapping = {"A": np.array([0,0]),
                            "C": np.array([0,1]),
                            "G": np.array([1,1]),
                            "T": np.array([1,0])}
        
        
    def cgr_encoding(self, dna_seq):
        prev_point = np.array([0.5,0.5])
        
        points = [prev_point]
        
        for a in dna_seq:
            if a not in self.cgt_mapping: continue
            point = 0.5 * (prev_point + self.cgt_mapping[a])
            points.append(point)
            prev_point = point
        
        return np.array(points)
        
    def __call__(self, genetic_string: str):
        
        """
        Args:
            genetics (str): The genetic data to be transformed.

        Returns:
            torch.Tensor: An n-by-2 list of points encoding the CGR.
        """
        
        # Truncate (padding does nothing to CGR)
        genetic_string = genetic_string[:self.length]

        return torch.from_numpy(self.cgr_encoding(genetic_string))
        
        


class GeneticDataset(Dataset):
    
    """
        A dataset class for the BIOSCAN genetic data. Samples are unpadded strings of nucleotides, including base pairs A, C, G, T and an unknown character N.

        Args:
            source (str): The path to the dataset file (csv or tsv).
            transform (callable, optional): Optional transforms to be applied to the genetic data. Default is None.
            drop_level (str): If supplied, the dataset will drop all rows where the given taxonomy level is not present. Default is None.
            allowed_classes ([(level, [class])]): If supplied, the dataset will only include rows where the given taxonomy level is within the given list of classes. Default is None. Use for validation and test sets.
            one_label (str): If supplied, the label will be the value of one_class
            classes: list[int]
            
        Returns:
            (genetics, label): A tuple containing the genetic data and the label (phylum, class, order, family, subfamily, tribe, genus, species, subspecies)
    """
    

    def __init__(self,
                 datapath: str,
                 transform='onehot',
                 level: str = None,
                 classes: list[str] = None
        ):
        
        self.data = pd.read_csv(datapath, sep="\t")
        self.level = level
        
        if transform == 'onehot':
            self.transform = GeneticOneHot()
        elif transform == 'kmer':
            self.transform = GeneticOneHot()
        elif transform == 'cgr':
            self.transform = GeneticCGR()
        else:
            self.transform = None
            
        
        self.taxnomy_level = ["phylum", "class", "order", "family", "subfamily", "tribe", "genus", "species", "subspecies"]


        if self.level:
            if not self.level in self.taxnomy_level:
                raise ValueError(f"drop_level must be one of {self.taxnomy_level}")
            self.data = self.data[self.data[self.level] != "not_classified"]

        if classes:
            self.classes = {
                c: i for i,c in enumerate(classes)
            }
        else:
            self.classes = {
                c: i for i,c in enumerate(self.get_classes(level)[0])
            }

    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        genetics = row["nucraw"]
        label = [row[c] for c in self.taxnomy_level]

        if self.transform:
            genetics = self.transform(genetics)

        label = label[self.taxnomy_level.index(self.level)]
        label = torch.tensor(self.classes[label])
            
        return genetics, label
    
    def __len__(self):
        return len(self.data)
    
    def get_classes(self, class_name: str):
        """Get a tuple of the list of the unique classes in the dataset, and their sizes for a given class name, e.x. order."""
        classes = self.data[class_name].unique()
        class_sizes = self.data[class_name].value_counts()

        return list(classes), list(class_sizes[classes])

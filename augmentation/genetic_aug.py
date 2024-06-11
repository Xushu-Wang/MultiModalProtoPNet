"""
This file augments a supplied dataset using the procedure
used in "Applying convolutional neural networks to speed up environmental DNA annotation in a highly diverse ecosystem"
by Flück et al.

They "added between zero and two random insertions and
deletions each, as well as noise in the form of a 5% mutation rate."

(Based on their code)
- Each insertion and deletion count are uniformly sampled
- Each insertion and deltion are size 1
- Each nucleotide has a 5% chance of being mutated after insertions and deletions.

Augmentation will be done in a balanced manner, such that the number of samples in each class is equal.

If the the number of samples for a class is greater than the maximum class size, the class will be downsampled to the maximum class size.
If they are less than the maximum class size, all samples will be included and the excess will be generated through augmentation.

Arguments:
- file: The path to the dataset to augment
- level: The level of the taxonomy to balance on
- out: The path to the output file (defaults to the input file with "_augmented" or "_augmented_balanced" appended)
- r: Noise rate (defaults to 0.05)
- sep: The separator used in the input file (defaults to "\t")
- samples: The number of samples to generate per class (defaults to the size of the maximum class) (ignored if folder mode is true)
- parent: The value of the parent taxonomy to balance on (defaults to None in which case all samples of the lower level are used) (ignored if folder mode is true)
- num_classes: The number of classes to generate (defaults to -1 in which case all classes are generated) (ignored if folder mode is true)
- folder_mode: Will generate a dataset from a tree of augmented images
- image_folder: Only does anything if folder_mode ios true
"""

import numpy as np
import pandas as pd
import argparse
from threading import Thread
import os
import shutil
import regex as re
from PIL import Image

def extract_id_with_extension(file):
    if "_" not in file:
        # This is a regular old run-of-the-mill path 
        return file
    # This is an augmented path
    pattern = r'(?<=_)(?<id>(BIOUG)?\d+(\-[^\.]\d+)?\.jpg)'
    return re.search(pattern, file).group('id')
    
def augment_sample(sample, r):
    insertion_count = np.random.randint(0, 3)
    deletion_count = np.random.randint(0, 3)

    insertion_indices = np.random.randint(0, len(sample), insertion_count)
    for idx in insertion_indices:
        sample = sample[:idx] + np.random.choice(list("ACGT")) + sample[idx:]
    
    deletion_indices = np.random.randint(0, len(sample), deletion_count)
    for idx in deletion_indices:
        sample = sample[:idx] + sample[idx+1:]
    
    mutation_indices = np.random.choice(len(sample), int(len(sample) * r), replace=False)
    for idx in mutation_indices:
        sample = sample[:idx] + np.random.choice(list("ACGT")) + sample[idx+1:]
    
    return sample

def augment_class(class_df, c, r, samples, out_list):
    if len(class_df) >= samples:
        output = class_df.sample(samples)
    else:
        remaining = samples - len(class_df)

        augmented_df = class_df.sample(remaining, replace=True, random_state=0)
        augmented_df = augmented_df.reset_index()
        augmented_df["sampleid"] = augmented_df["sampleid"].astype(str) + "_augmented_"  + augmented_df.index.astype(str) 
        augmented_df["nucraw"] = augmented_df["nucraw"].apply(lambda x: augment_sample(x, r))
        output = pd.concat([class_df, augmented_df])

    print(f"Class {c} Augmented {len(class_df)} -> {len(output)}")
    out_list.append(output)


def augment(df, level, r, samples):
    classes = df[level].unique()

    outputs = []

    for c in classes:
        class_df = df[df[level] == c]
        augment_class(class_df, c, r, samples, outputs)

    return pd.concat(outputs)      
        
def get_total_balanced_size(df, level):
    return len(df[level].unique()) * df[level].value_counts().max()

def process_df(df, level, parent, parent_type, num_classes):
    levels = ["phylum", "class", "order", "family", "subfamily", "tribe", "genus", "species", "subspecies"]
    
    df = df[df[level] != "not_classified"]
    
    if level not in levels[1:]:
        raise ValueError(f"level must be one of {levels[1:]}")
    
    if parent is not None:
        df = df[df[parent_type] == parent]
    
    if num_classes != -1:
        class_counts = df[level].value_counts()
        classes = class_counts.index[:num_classes]
        df = df[df[level].isin(classes)]
    
    return df

def folder_augment(df, image_folder, r):
    out_df = pd.DataFrame(columns=[*df.columns])
    df = df.set_index("image_file")
    rows = []

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"{image_folder} not found")

    tot = 0
    err = 0

    for path, _, files in os.walk(image_folder):
        for file in files:
            if ".jpg" not in file:
                continue
            extracted_id = extract_id_with_extension(file)
            # Try to find the matching ID in the dataframe
            tot += 1
            try:
                matching_row = df.loc[extracted_id]
                print(matching_row)
                exit()
                obj = matching_row.to_dict()
                obj["nucraw"] = augment_sample(obj["nucraw"], r)
                obj["image_path"] = file
                rows.append(obj)
            except Exception as e:
                err += 1
                print(f"Could not find matching sample for {extracted_id} - {os.path.join(path, file)}")
                exit()

    out_df = pd.DataFrame(rows)

    print(f"Error Rate: {err / tot * 100:.2f}%")

    return out_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="Path to the dataset to augment")
    parser.add_argument("level", type=str, help="The level of the taxonomy to balance on")
    parser.add_argument('--out', type=str, default=None, help="Path to the output file (don't include _augmented)")
    parser.add_argument('-r', type=float, default=0.05, help="Noise rate")
    parser.add_argument('--sep', type=str, default="\t", help="The separator used in the input file")
    parser.add_argument('--samples', type=int, default=-1, help="The number of samples to generate per class (defaults to the maximum class size)")
    parser.add_argument('--parent_type', type=str, default=None, help="Only keep samples with this parent type = parent")
    parser.add_argument('--parent', type=str, default=None, help="The value of the parent taxonomy to balance on")
    parser.add_argument('--num_classes', type=int, default=-1, help="The number of classes to generate (defaults to all classes)")
    parser.add_argument('--folder_mode', action="store_true", help="Will generate a dataset from a tree of augmented images")
    parser.add_argument('--image_folder', type=str, default=None, help="Only does anything if folder_mode is true")
    args = parser.parse_args()

    np.random.seed(0)

    df = pd.read_csv(args.file, sep=args.sep)

    if args.folder_mode:
        out_df = folder_augment(df, args.image_folder, args.r)
    else:
        df = process_df(df, args.level, args.parent, args.parent_type, args.num_classes)
        if args.out is None:
            args.out = args.file

        if ".tsv" in args.out:
            chopped_out = args.out.replace(".tsv", "_chopped.tsv")
            args.out = args.out.replace(".tsv", "_augmented.tsv")
        else:
            chopped_out = args.out.replace(".csv", "_chopped.csv")
            args.out = args.out.replace(".csv", "_augmented.csv")

        if args.parent:
            df.to_csv(chopped_out, sep=args.sep, index=False)
            print(f"Chopped dataset written to {chopped_out}")

        if args.samples == -1:
            args.samples = df[args.level].value_counts().max()

        print(f"Input Size: {len(df):,}")
        print(f"Output size: {args.samples * len(df[args.level].unique()):,}")
        cont = input("Continue? (Y/n) ")

        if cont.lower() == "n":
            exit()

        out_df = augment(df, args.level, args.r, args.samples)
    
    out_df.to_csv(args.out, sep=args.sep, index=False)
    print(f"Output written to {args.out}")
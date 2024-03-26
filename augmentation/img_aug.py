import Augmentor
import os
import argparse
from utils.helpers import makedir
        
        
def img_augmentation(root_dir):

    """
    Image Augmentation for preprocessing caltech bird (cub) dataset
    
    Default operation includes rotation, skew, shear, and random distortion
    """
    
    datasets_root_dir = root_dir     # Default Directory is './datasets/cub200_cropped/'
    dir = datasets_root_dir + 'train_cropped/'
    target_dir = datasets_root_dir + 'train_cropped_augmented/'

    makedir(target_dir)
    folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
    target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

    for i in range(len(folders)):
        fd = folders[i]
        tfd = target_folders[i]
        # rotation
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # skew
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.skew(probability=1, magnitude=0.2)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # shear
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        
        #random_distortion
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        
    print("Image Augmentation, Done")
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_dir', type=str, default='./datasets/cub200_cropped/', 
                        help='Root directory containing the dataset')
    
    args = parser.parse_args()
    
    root_dir = args.root_dir
    
    img_augmentation(root_dir)
    

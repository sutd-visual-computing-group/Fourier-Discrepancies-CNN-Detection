# General modules
import os, math, argparse
from itertools import islice
from typing import Any, List, Dict, Union, Optional

# Imaging/ scientific modules
import numpy as np
from PIL import Image 

# Pytorch modules
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Webdataset module (https://github.com/tmbdev/webdataset)
import webdataset as wds

# Lightning module
import pytorch_lightning as pl

# Other modules
from tqdm import tqdm

# Importing our own modules here
from settings import NUM_TRAINING_FILES



def celeba_samples( file_paths: List , 
                    crop_size: int = 178):
    """
    Create CelebaA samples by center cropping into 178x178
    
    Args:
        file_paths  : List of file paths of CelebA images
        crop_size   : Crop size of image
    """
    y1, y2, x1, x2 = crop_celeba_image_coords(crop_size)

    for fname in file_paths:
        image = np.array(Image.open(fname))[y1:y2, x1:x2, :]

        sample = {
            "__key__" : fname.split('/')[-1].split('.')[0],
            "png" : image
        }
        yield sample


def lsun_samples(   file_paths: List , 
                    crop_size: int = 256):
    """
    Create LSUN samples by center cropping into 256x256
    
    Args:
        file_paths  : List of file paths of LSUN Bedroom images
        crop_size   : Crop size of image
    """
    for fname in file_paths:
        image = np.array(Image.open(fname))
        height, width = image.shape[0], image.shape[1]
        y1, y2, x1, x2 = crop_lsun_image_coords(height, width, crop_size)
        image = image[y1:y2, x1:x2, :]

        sample = {
            "__key__" : fname.split('/')[-1].split('.')[0],
            "png" : image
        }
        yield sample


def crop_celeba_image_coords(crop_size: int):
    """
    Get coordinates for cropping images (y1, y2, x1, x2)
    The original images are 218x178

    Args:
        crop_size   : Crop size of image

    """
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    return offset_height, offset_height + crop_size, offset_width, offset_width + crop_size


def crop_lsun_image_coords( height: int, 
                            width: int, 
                            crop_size: int):
    """
    Get coordinates for cropping images (y1, y2, x1, x2)
    The original images are either (x, 256) or (256, x) where x>= 256

    Args:
        crop_size   : Crop size of image
    """
    offset_height = (height - crop_size) // 2
    offset_width = (width - crop_size) // 2
    return offset_height, offset_height + crop_size, offset_width, offset_width + crop_size


def create_tars(    img_dir: str, 
                    save_dir: str, 
                    max_tar: int, 
                    tar_save_suffix: str, 
                    num_files: int):
    """
    Use tar files to allow for better I/O. 
    Combine with sharding to yield the benefits of parallel computing.
    
    Args:
        img_dir         : Directory containing images
        save_dir        : Directory to save tar files
        max_tar         : Maximum number of tar files (prefer multiple of 8)
        tar_save_suffix : Suffix to append to tar files (use dataset name)
        num_files       : Number of files in the img_dir to create tar files
    """
    # Get the training dataset
    file_paths = [ os.path.join(img_dir, i) for i in os.listdir(img_dir) ]
    file_paths.sort()
    file_paths = file_paths[:num_files]
    tar_save_dir = os.path.join(save_dir, tar_save_suffix)
    os.makedirs(tar_save_dir, exist_ok=True)

    # Decide maximum files per tar
    maximum_files_per_tar = int(math.ceil( len(file_paths)/max_tar ))

    # Get samples map fn based on tar_save_suffix
    sample_fn = celeba_samples if tar_save_suffix=='celeba' else lsun_samples

    with wds.ShardWriter( os.path.join(tar_save_dir, '{}-%06d.tar'.format(tar_save_suffix)), maxcount=maximum_files_per_tar ) as sink:
        with tqdm(total=len(file_paths)) as pbar:
            for sample in islice(sample_fn(file_paths), 0, len(file_paths)):
                sink.write(sample)
                pbar.update(1)
    

def load_webdataset_for_gan(    url: str, 
                                resize: int, 
                                image_transform: Optional, 
                                num_files: int):
    """
    Load webdataset tars for GAN training. 
    
    Args:
        url             : url / locations indicating the tar files
        resize          : Size of images used in GAN training (Our work uses 128 x 128 images)
        image_transform : Pytorch image transforms
        num_files       : Number of files to retrieve from the tars
    """
    dataset = (wds.WebDataset(url, length=num_files)
                .decode("pil")
                .to_tuple("png")
                .map_tuple(image_transform, lambda x: x)
                .shuffle(1000)
                )

    return dataset



class CelebADataModule(pl.LightningDataModule):
    """
    Pytorch Lightning Data module for CelebA dataset.
    Refer to https://github.com/PyTorchLightning/pytorch-lightning/blob/bb9ace43334ad50e3758d9cff08ad34216c7d4da/pytorch_lightning/core/datamodule.py for more details.
    """
    
    def __init__(self, 
                url: str ,
                resize: int, 
                image_transform, 
                num_files: int,
                batch_size: int, 
                num_workers: int):
        super().__init__()
        self.url = url
        self.resize = resize
        self.image_transform = image_transform
        self.num_files = num_files

        self.batch_size = batch_size
        self.num_workers = num_workers


    def prepare_data(self):
        pass


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders. We only need train dataloaders
        if stage == 'fit' or stage is None:
            self.celeba_webdataset = load_webdataset_for_gan(self.url, self.resize, self.image_transform, self.num_files)


    def train_dataloader(self):
        return DataLoader(self.celeba_webdataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                            drop_last=True,
                            pin_memory=True)



def main():
    # > Setup command line arguments
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--tar_save_dir', required=True)
    parser.add_argument('--max_tar', default=8, type=int)
    parser.add_argument('--dataset_name', required=True, choices=['celeba', 'lsun'])
    args = parser.parse_args()
    
    # Create dataset
    create_tars( args.img_dir, args.tar_save_dir, args.max_tar, args.dataset_name, NUM_TRAINING_FILES[args.dataset_name] )



if __name__ == "__main__":
    main()
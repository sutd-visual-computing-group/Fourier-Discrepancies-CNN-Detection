# General modules
import os

# Pytorch modules
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Webdataset module (https://github.com/tmbdev/webdataset)
import webdataset as wds

# Lightning module
import pytorch_lightning as pl

# Importing our own modules here
from dataset_tool import load_webdataset_for_gan



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
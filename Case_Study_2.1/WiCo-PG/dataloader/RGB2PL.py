import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import scipy.io
import numpy as np

class RGB2PLDataset(Dataset):
    """A custom dataset for loading images from a single folder (without categories)."""

    def __init__(self, data_dir, data_dir2, transform=None, transform2=None, num_files=1254):
        self.data_dir = data_dir
        self.transform = transform
        self.transform2 = transform2
        self.image_RGB_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_pl_paths = [os.path.join(data_dir2, fname) for fname in os.listdir(data_dir) if
                                fname.endswith(('.png', '.jpg', '.jpeg'))]



    def __len__(self):
        return len(self.image_RGB_paths)



    def __getitem__(self, idx):
        image_path_RGB = self.image_RGB_paths[idx]
        image_path_pl = self.image_pl_paths[idx]
        #print(image_path_RGB, image_path_pl)
        image_RGB = Image.open(image_path_RGB).convert("RGB")  # Open image and convert to RGB
        # label = 0  # Since all images belong to the same class (no categories)
        image_pl = Image.open(image_path_pl).convert("L")
        # print(type(image)) #<class 'PIL.Image.Image'>

        if self.transform:
            image_RGB = self.transform(image_RGB)  # Apply transformations (e.g., resize, normalize)
        if self.transform2:
            image_pl = self.transform2(image_pl)
        # print(type(image))  # <class 'PIL.Image.Image'>
        #print(image_pl.shape, image_RGB.shape)

        return image_RGB, image_pl
    

class RGBD2PLDataset(Dataset):
    """A custom dataset for loading images from a single folder (without categories)."""

    def __init__(self, data_dir, data_dir2, data_dir3, transform=None, transform2=None,  num_files=1254):
        self.data_dir = data_dir
        self.data_dir2 = data_dir2
        self.data_dir3 = data_dir3
        self.transform = transform
        self.transform2 = transform2
   
        self.image_RGB_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_dep_paths = [os.path.join(data_dir2, fname) for fname in os.listdir(data_dir2) if
                                fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_pl_paths = [os.path.join(data_dir3, fname) for fname in os.listdir(data_dir3) if
                                fname.endswith(('.png', '.jpg', '.jpeg'))]



    def __len__(self):
        return len(self.image_pl_paths)



    def __getitem__(self, idx):
        image_path_RGB = self.image_RGB_paths[idx]
        image_path_dep = self.image_dep_paths[idx]
        image_path_pl = self.image_pl_paths[idx]
        #print(image_path_RGB, image_path_pl)
        image_RGB = Image.open(image_path_RGB).convert("RGB")  # Open image and convert to RGB
        # label = 0  # Since all images belong to the same class (no categories)
        image_dep = Image.open(image_path_dep).convert("L")
        image_pl = Image.open(image_path_pl).convert("L")
        # print(type(image)) #<class 'PIL.Image.Image'>

        if self.transform:
            image_RGB = self.transform(image_RGB)  # Apply transformations (e.g., resize, normalize)
        if self.transform2:
            image_dep = self.transform2(image_dep)
        if self.transform2:
            image_pl = self.transform2(image_pl)
        
        image_RGBD = torch.cat((image_RGB, image_dep), dim=0)
        # print(type(image))  # <class 'PIL.Image.Image'>
        #print(image_pl.shape, image_RGB.shape)

        return image_RGBD, image_pl
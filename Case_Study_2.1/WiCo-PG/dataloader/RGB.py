import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class CustomImageDataset(Dataset):
    """A custom dataset for loading images from a single folder (without categories)."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
        # label = 0  # Since all images belong to the same class (no categories)

        # print(type(image)) #<class 'PIL.Image.Image'>

        if self.transform:
            image = self.transform(image)  # Apply transformations (e.g., resize, normalize)

        # print(type(image))  # <class 'PIL.Image.Image'>

        return image  # Return image and its label


class CustomPLDataset(Dataset):
    """A custom dataset for loading images from a single folder (without categories)."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")  # Open image and convert to RGB
        # label = 0  # Since all images belong to the same class (no categories)

        # print(type(image)) #<class 'PIL.Image.Image'>

        if self.transform:
            image = self.transform(image)  # Apply transformations (e.g., resize, normalize)

        # print(type(image))  # <class 'PIL.Image.Image'>

        return image  # Return image and its label

class RGBDDataset(Dataset):
    """A custom dataset for loading images from a single folder (without categories)."""

    def __init__(self, rgb_dir, dep_dir,  rgb_transform=None, dep_transform=None):
        self.rgb_dir = rgb_dir
        self.dep_dir = dep_dir
        self.rgb_transform = rgb_transform
        self.dep_transform = dep_transform
        self.rgb_image_paths = [os.path.join(rgb_dir, fname) for fname in os.listdir(rgb_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.dep_image_paths = [os.path.join(dep_dir, fname) for fname in os.listdir(dep_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.rgb_image_paths)

    def __getitem__(self, idx):
        rgb_image_path = self.rgb_image_paths[idx]
        dep_image_path = self.dep_image_paths[idx]
        rgb_image = Image.open(rgb_image_path).convert("RGB")  # Open image and convert to RGB
        dep_image = Image.open(dep_image_path).convert("L")
        

        # label = 0  # Since all images belong to the same class (no categories)

        # print(type(image)) #<class 'PIL.Image.Image'>

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image) 
        if self.dep_transform:
            dep_image = self.dep_transform(dep_image)
        image = torch.cat((rgb_image, dep_image), dim=0) # Apply transformations (e.g., resize, normalize)

        # print(type(image))  # <class 'PIL.Image.Image'>

        return image  # Return image and its label


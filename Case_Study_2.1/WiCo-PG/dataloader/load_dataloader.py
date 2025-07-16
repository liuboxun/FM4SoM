# Importing Libraries
import torch

from dataloader import load_mnist, load_cifar10
from dataloader.RGB import CustomImageDataset, CustomPLDataset, RGBDDataset
from dataloader.RGB2PL import RGB2PLDataset, RGBD2PLDataset

import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


def load_dataloader(
    name: str = "mnist",
    batch_size: int = 2,
    image_size: int = 256,
    num_workers: int = 4,
    save_path: str = "data",
) -> torch.utils.data.DataLoader:
    """Load the data loader for the given name.

    Args:
        name (str, optional): The name of the data loader. Defaults to "mnist".
        batch_size (int, optional): The batch size. Defaults to 2.
        image_size (int, optional): The image size. Defaults to 256.
        num_workers (int, optional): The number of workers to use for the dataloader. Defaults to 4.
        save_path (str, optional): The path to save the data to. Defaults to "data".

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """

    if name == "mnist":
        return load_mnist(
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            save_path=save_path,
        )

    elif name == "cifar10":
        return load_cifar10(
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            save_path=save_path,
        )
#### 微调的数据集RGBD、pl、RGBD2PL
###63-50_400s
    elif name == "RGB_D_50_400s":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_400/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_400/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "pl_50_400s":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_400/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGBD2PL_50_400s":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_400/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_400/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_400/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    
    elif name == "RGB2PL_70_15ghz":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/dataset_kaiyuan//"
        dep_dir = "/home/smr/smr_base_model/VQGAN/data/dataset_kaiyuan/dep/"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/dataset_kaiyuan/dep"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    
    elif name == "RGB2PL_70_kk":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/dataset_kaiyuan/RGB/"
       
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/dataset_kaiyuan/pl/"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader



    ###63-50_500s
    elif name == "RGB_D_50_500s":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_500/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_500/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "pl_50_500s":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_500/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGBD2PL_50_500s":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_500/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_500/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_500/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    


    ###63-50_300s
    elif name == "RGB_D_50_300s":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_300/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_300/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "pl_50_300s":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_300/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGBD2PL_50_300s":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_300/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_300/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_300/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    ###63-50_200s
    elif name == "RGB_D_50_200s":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_200/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_200/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "pl_50_200s":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_200/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGBD2PL_50_200s":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_200/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_200/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_200/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    

    ###63-50_100s
    elif name == "RGB_D_50_100s":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_100/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_100/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "pl_50_100s":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_100/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGBD2PL_50_100s":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_100/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_100/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_100/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    

    ###63-50_50s
    elif name == "RGB_D_50_50s":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_50/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_50/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "pl_50_50s":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_50/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGBD2PL_50_50s":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_50/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_50/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_50/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    
    ###63-50_40s
    elif name == "RGB_D_50_40s":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_40/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_40/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "pl_50_40s":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_40/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGBD2PL_50_40s":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_40/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_40/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_40/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    ###63-50_30s
    elif name == "RGB_D_50_30s":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_30/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_30/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "pl_50_30s":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_30/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGBD2PL_50_30s":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_30/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_30/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train_30/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB_63_input_height":
        """Load a custom dataset with no categories (all images in a single folder)."""

        data_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomImageDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader


    elif name == "RGB_63":
        """Load a custom dataset with no categories (all images in a single folder)."""

        data_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomImageDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader

    elif name == "RGB_D_63":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/train/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/train/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader

    elif name == "RGB_50":
        """Load a custom dataset with no categories (all images in a single folder)."""

        data_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomImageDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "RGB_D_50":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
    elif name == "RGB_D_70":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/70m_full/train/RGB"
        dep_data_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/70m_full/train/dep"

        # Define the transform for the images
        rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dep_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = RGBDDataset(rgb_dir=rgb_data_dir, dep_dir=dep_data_dir, rgb_transform=rgb_transform, dep_transform=dep_transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
   

    elif name == "RGB_70":
        """Load a custom dataset with no categories (all images in a single folder)."""

        data_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomImageDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader

    elif name == "RGB_80":
        """Load a custom dataset with no categories (all images in a single folder)."""

        data_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomImageDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader

    elif name == "RGB_multi_height_50+63+70":
        """Load a custom dataset with no categories (all images in a single folder)."""
        data_dir_50m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        data_dir_63m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        data_dir_70m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        data_dir_80m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomImageDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomImageDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomImageDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomImageDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_50m + dataset_63m + dataset_70m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader

    elif name == "RGB_multi_height_50+63+80":
        """Load a custom dataset with no categories (all images in a single folder)."""
        data_dir_50m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        data_dir_63m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        data_dir_70m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        data_dir_80m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomImageDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomImageDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomImageDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomImageDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_50m + dataset_63m + dataset_80m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader

    elif name == "RGB_multi_height_50+70+80":
        """Load a custom dataset with no categories (all images in a single folder)."""
        data_dir_50m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        data_dir_63m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        data_dir_70m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        data_dir_80m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomImageDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomImageDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomImageDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomImageDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_50m + dataset_70m + dataset_80m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader

    elif name == "RGB_multi_height_63+70+80":
        """Load a custom dataset with no categories (all images in a single folder)."""
        data_dir_50m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        data_dir_63m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        data_dir_70m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        data_dir_80m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomImageDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomImageDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomImageDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomImageDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_63m + dataset_70m + dataset_80m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader


    elif name == "RGB_all_height":
        """Load a custom dataset with no categories (all images in a single folder)."""
        data_dir_50m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        data_dir_63m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        data_dir_70m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        data_dir_80m = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomImageDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomImageDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomImageDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomImageDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_50m + dataset_63m + dataset_70m + dataset_80m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader


    elif name == "pl_50":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    
   

    elif name == "pl_63":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/train/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "pl_70":
        # 创建数据集和数据加载器
        data_dir = '/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/70m_full/train/pl'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "pl_80":
        # 创建数据集和数据加载器
        data_dir = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset = CustomPLDataset(data_dir=data_dir, transform=transform)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "pl_multi_height_50+63+70":
        # 创建数据集和数据加载器
        data_dir_50m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50'
        data_dir_63m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63'
        data_dir_70m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70'
        data_dir_80m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80'
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomPLDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomPLDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomPLDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomPLDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_50m + dataset_63m + dataset_70m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "pl_multi_height_50+63+80":
        # 创建数据集和数据加载器
        data_dir_50m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50'
        data_dir_63m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63'
        data_dir_70m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70'
        data_dir_80m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80'
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomPLDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomPLDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomPLDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomPLDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_50m + dataset_63m + dataset_80m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "pl_multi_height_50+70+80":
        # 创建数据集和数据加载器
        data_dir_50m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50'
        data_dir_63m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63'
        data_dir_70m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70'
        data_dir_80m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80'
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomPLDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomPLDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomPLDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomPLDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_50m + dataset_70m + dataset_80m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "pl_multi_height_63+70+80":
        # 创建数据集和数据加载器
        data_dir_50m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50'
        data_dir_63m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63'
        data_dir_70m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70'
        data_dir_80m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80'
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomPLDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomPLDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomPLDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomPLDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_63m + dataset_70m + dataset_80m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )


    elif name == "pl_all_height":
        # 创建数据集和数据加载器
        data_dir_50m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50'
        data_dir_63m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63'
        data_dir_70m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70'
        data_dir_80m = '/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80'
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])

        # Create the dataset from the folder of images
        dataset_50m = CustomPLDataset(data_dir=data_dir_50m, transform=transform)
        dataset_63m = CustomPLDataset(data_dir=data_dir_63m, transform=transform)
        dataset_70m = CustomPLDataset(data_dir=data_dir_70m, transform=transform)
        dataset_80m = CustomPLDataset(data_dir=data_dir_80m, transform=transform)
        dataset = dataset_50m + dataset_63m + dataset_70m + dataset_80m

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGB2PL":
        """Load a custom dataset with no categories (all images in a single folder)."""
        data_dir = "/home/smr/smr_base_model/VQGAN/data/my_data/RGB_63m"
        # 创建数据集和数据加载器
        folder_path = "/home/smr/smr_base_model/VQGAN/data/my_data/pl_63m"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=data_dir, data_dir2=folder_path, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_70":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_50":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_63":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    
    elif name == "RGBD2PL_63":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/train/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/train/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/train/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    

    elif name == "RGBD2PL_50":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/train/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGBD2PL_70":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/70m_full/train/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/70m_full/train/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/70m_full/train/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
        
    
    


    elif name == "RGB2PL_80":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_80_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/test/RGB_80"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/test/pl_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    
    elif name == "RGBD2PL_63_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/test/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/test/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/test/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    
    elif name == "RGBD2PL_50_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/test/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/test/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/50_28/test/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader
    
    elif name == "RGBD2PL_70_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/70m_full/test/RGB"
        dep_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/70m_full/test/dep"
        # 创建数据集和数据加载器
        pl_dir = "/mnt/wwn-0x5000c500f6ba760d/SMR_dataset/70m_full/test/pl"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGBD2PLDataset(data_dir=rgb_dir, data_dir2=dep_dir, data_dir3=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_70_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_28_multi_height/test/RGB_70"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_28_multi_height/test/pl_70"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_70_1.6ghz_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_70_multi_fc/test/RGB_70"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_70_multi_fc/test/pl_70_1.6"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_70_5.9ghz_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_70_multi_fc/5.9ghz/test/RGB_70"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_70_multi_fc/5.9ghz/test/pl_70_5.9"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_70_15ghz_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_70_multi_fc/15ghz/test/RGB_70"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_70_multi_fc/15ghz/test/pl_70_15"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_63_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/test/RGB_63"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/test/pl_63"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_50_test":
        """Load a custom dataset with no categories (all images in a single folder)."""
        rgb_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/test/RGB_50"
        # 创建数据集和数据加载器
        pl_dir = "/home/smr/smr_base_model/VQGAN/data/data_random/test/pl_50"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])
        # Create the dataset from the folder of images
        dataset = RGB2PLDataset(data_dir=rgb_dir, data_dir2=pl_dir, transform=transform, transform2=transform2)
        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_multi_height_50+63+70":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        pl_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50"

        rgb_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        pl_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63"

        rgb_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        pl_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70"

        rgb_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        pl_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80"

        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])


        # Create the dataset from the folder of images
        dataset_50m = RGB2PLDataset(data_dir=rgb_dir_50, data_dir2=pl_dir_50, transform=transform, transform2=transform2)
        dataset_63m = RGB2PLDataset(data_dir=rgb_dir_63, data_dir2=pl_dir_63, transform=transform, transform2=transform2)
        dataset_70m = RGB2PLDataset(data_dir=rgb_dir_70, data_dir2=pl_dir_70, transform=transform, transform2=transform2)
        dataset_80m = RGB2PLDataset(data_dir=rgb_dir_80, data_dir2=pl_dir_80, transform=transform, transform2=transform2)

        dataset = dataset_50m +dataset_63m + dataset_70m
        #print('dataset',dataset[])

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_multi_height_50+70+80":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        pl_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50"

        rgb_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        pl_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63"

        rgb_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        pl_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70"

        rgb_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        pl_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])


        # Create the dataset from the folder of images
        dataset_50m = RGB2PLDataset(data_dir=rgb_dir_50, data_dir2=pl_dir_50, transform=transform, transform2=transform2)
        dataset_63m = RGB2PLDataset(data_dir=rgb_dir_63, data_dir2=pl_dir_63, transform=transform, transform2=transform2)
        dataset_70m = RGB2PLDataset(data_dir=rgb_dir_70, data_dir2=pl_dir_70, transform=transform, transform2=transform2)
        dataset_80m = RGB2PLDataset(data_dir=rgb_dir_80, data_dir2=pl_dir_80, transform=transform, transform2=transform2)

        dataset = dataset_50m + dataset_70m + dataset_80m
        #print('dataset',dataset[])

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_multi_height_50+63+80":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        pl_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50"

        rgb_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        pl_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63"

        rgb_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        pl_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70"

        rgb_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        pl_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])


        # Create the dataset from the folder of images
        dataset_50m = RGB2PLDataset(data_dir=rgb_dir_50, data_dir2=pl_dir_50, transform=transform, transform2=transform2)
        dataset_63m = RGB2PLDataset(data_dir=rgb_dir_63, data_dir2=pl_dir_63, transform=transform, transform2=transform2)
        dataset_70m = RGB2PLDataset(data_dir=rgb_dir_70, data_dir2=pl_dir_70, transform=transform, transform2=transform2)
        dataset_80m = RGB2PLDataset(data_dir=rgb_dir_80, data_dir2=pl_dir_80, transform=transform, transform2=transform2)

        dataset = dataset_50m +dataset_63m + dataset_80m
        #print('dataset',dataset[])

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_multi_height_63+70+80":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        pl_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50"

        rgb_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        pl_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63"

        rgb_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        pl_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70"

        rgb_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        pl_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])


        # Create the dataset from the folder of images
        dataset_50m = RGB2PLDataset(data_dir=rgb_dir_50, data_dir2=pl_dir_50, transform=transform, transform2=transform2)
        dataset_63m = RGB2PLDataset(data_dir=rgb_dir_63, data_dir2=pl_dir_63, transform=transform, transform2=transform2)
        dataset_70m = RGB2PLDataset(data_dir=rgb_dir_70, data_dir2=pl_dir_70, transform=transform, transform2=transform2)
        dataset_80m = RGB2PLDataset(data_dir=rgb_dir_80, data_dir2=pl_dir_80, transform=transform, transform2=transform2)

        dataset = dataset_63m + dataset_70m + dataset_80m
        #print('dataset',dataset[])

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader


    elif name == "RGB2PL_all_height":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_50"
        pl_dir_50 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_50"

        rgb_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_63"
        pl_dir_63 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_63"

        rgb_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_70"
        pl_dir_70 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_70"

        rgb_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/RGB_80"
        pl_dir_80 = "/home/smr/smr_base_model/VQGAN/data/data_random/train/pl_80"
        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])


        # Create the dataset from the folder of images
        dataset_50m = RGB2PLDataset(data_dir=rgb_dir_50, data_dir2=pl_dir_50, transform=transform, transform2=transform2)
        dataset_63m = RGB2PLDataset(data_dir=rgb_dir_63, data_dir2=pl_dir_63, transform=transform, transform2=transform2)
        dataset_70m = RGB2PLDataset(data_dir=rgb_dir_70, data_dir2=pl_dir_70, transform=transform, transform2=transform2)
        dataset_80m = RGB2PLDataset(data_dir=rgb_dir_80, data_dir2=pl_dir_80, transform=transform, transform2=transform2)

        dataset = dataset_50m + dataset_63m + dataset_70m + dataset_80m
        #print('dataset',dataset[])

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    elif name == "RGB2PL_multi_height_63+80":
        """Load a custom dataset with no categories (all images in a single folder)."""

        rgb_dir_63 = "/home/smr/smr_base_model/VQGAN/data/my_data/RGB_63m"
        pl_dir_63 = "/home/smr/smr_base_model/VQGAN/data/my_data/pl_63m"

        rgb_dir_70 = "/home/smr/smr_base_model/VQGAN/data/my_data/RGB_70m"
        pl_dir_70 = "/home/smr/smr_base_model/VQGAN/data/my_data/pl_70m"

        rgb_dir_80 = "/home/smr/smr_base_model/VQGAN/data/my_data/RGB_80m"
        pl_dir_80 = "/home/smr/smr_base_model/VQGAN/data/my_data/pl_80m"

        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])
        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the images
        ])


        # Create the dataset from the folder of images
        dataset_63m = RGB2PLDataset(data_dir=rgb_dir_63, data_dir2=pl_dir_63, transform=transform, transform2=transform2)
        dataset_70m = RGB2PLDataset(data_dir=rgb_dir_70, data_dir2=pl_dir_70, transform=transform, transform2=transform2)
        dataset_80m = RGB2PLDataset(data_dir=rgb_dir_80, data_dir2=pl_dir_80, transform=transform, transform2=transform2)

        dataset = dataset_80m + dataset_63m
        #print('dataset',dataset[])

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )


        # 遍历数据集
        '''for batch_idx, (rgb_images, pl_images) in enumerate(dataloader):
            # rgb_images 包含 RGB 图像，pl_images 包含单通道图像
            # 注意：rgb_images 和 pl_images 的形状和内容取决于你的 dataset 和 transform

            # 打印批次索引和图像形状
            print(f"Batch {batch_idx + 1}:")
            print(f"  RGB Image Shape: {rgb_images.shape}")
            print(f"  PL Image Shape: {pl_images.shape}")'''



        return dataloader
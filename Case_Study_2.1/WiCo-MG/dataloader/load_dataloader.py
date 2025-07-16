import torch
from dataloader import DoADataset, RGB2DoADataset

from torch.utils.data import DataLoader
from torchvision import transforms



def load_dataloader(
    name: str = "mnist",
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    save_path: str = "data",
    num_files: int = 10,
) -> torch.utils.data.DataLoader:


    if name == "DoA":
        # 创建数据集和数据加载器
        folder_path = "./data"

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            # transforms.ToTensor(),  # Convert images to PyTorch tensors
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        dataset = DoADataset(folder_path, transform=transform, num_files=num_files)

        print('dataset',dataset)


        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader


    elif name == "RGB2DoA":
        """Load a custom dataset with no categories (all images in a single folder)."""

        data_dir = "./data"
        folder_path = "./data"

        # Define the transform for the images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the images
        ])

        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to the target size
        ])

        # Create the dataset from the folder of images
        dataset = RGB2DoADataset(data_dir=data_dir, data_dir2=folder_path, transform=transform, transform2=transform2, num_files=num_files)

        # Create a DataLoader for the dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    


import os
import torch

from torch.utils.data import Dataset

from PIL import Image
import scipy.io
import numpy as np


def numeric_sort(file_name):

    return int(''.join(filter(str.isdigit, file_name)))

class RGB2DoADataset(Dataset):
    """A custom dataset for loading images from a single folder (without categories)."""

    def __init__(self, data_dir, data_dir2, transform=None, transform2=None, num_files=10):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f'image{i}.png' for i in range(1, num_files + 1)]

        self.files = [f'time{i}_angle_data_path1_phi.mat' for i in range(1, num_files + 1)]
        self.folder_path = data_dir2
        self.transform2 = transform2

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_dir, self.image_files[idx])

        image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
        image = image.rotate(270, expand=True)

        if self.transform:
            image = self.transform(image)  # Apply transformations (e.g., resize, normalize)


        # 读取.mat文件
        mat_file = os.path.join(self.folder_path, self.files[idx])
        data = scipy.io.loadmat(mat_file)

        # 根据文件名生成变量名
        input_time_var = f'angle_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        DoA = np.float32(input_time)

        DoA = np.fliplr(DoA).copy()

        data_tensor = torch.tensor(DoA)
        data_tensor = data_tensor.unsqueeze(0)

        if self.transform2:
            data_tensor = self.transform2(data_tensor)  # Apply transformations (e.g., resize, normalize)
            data_tensor = data_tensor / 360.0
            mean = 0.5
            std = 0.5
            data_tensor = (data_tensor - mean) / std

        return image, data_tensor
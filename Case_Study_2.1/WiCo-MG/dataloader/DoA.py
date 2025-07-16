import torch
from torch.utils.data import Dataset
import scipy.io
import os
import numpy as np



class DoADataset(Dataset):
    def __init__(self, folder_path, num_files=10, transform=None):

        self.files = [f'time{i}_angle_data_path1_phi.mat' for i in range(1, num_files + 1)]
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 读取.mat文件
        mat_file = os.path.join(self.folder_path, self.files[idx])
        data = scipy.io.loadmat(mat_file)

        input_time_var = f'angle_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        DoA = np.float32(input_time)

        DoA = np.fliplr(DoA).copy()

        data_tensor=torch.tensor(DoA)
        data_tensor = data_tensor.unsqueeze(0)

        if self.transform:
            data_tensor = self.transform(data_tensor)  # Apply transformations (e.g., resize, normalize)
            data_tensor = data_tensor / 360.0
            mean = 0.5
            std = 0.5
            data_tensor = (data_tensor - mean) / std

        return data_tensor
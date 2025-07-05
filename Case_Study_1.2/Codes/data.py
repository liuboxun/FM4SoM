import os.path
import os
import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from einops import rearrange
from numpy import random
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader

def LoadBatch(H):
    # H: ...     [tensor complex]
    # out: ..., 2  [tensor real]
    size = list(H.shape)
    H_real = np.zeros(size + [2])
    H_real[..., 0] = H.real
    H_real[..., 1] = H.imag
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def Transform_TDD_FDD(H, Nt=4, Nr=4):
    # H: B,T,mul    [tensor real]
    # out:B',Nt,Nr  [tensor complex]
    H = H.reshape(-1, Nt, Nr, 2)
    H_real = H[..., 0]
    H_imag = H[..., 1]
    out = torch.complex(H_real, H_imag)
    return out


def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise


# SNR 5, 15, 20
class Dataset(data.Dataset):
    def __init__(self, file_path, is_train=1, ir=1, SNR=25, prev_len=32, pred_len=32, num_pilot=8,
                 is_show=1):
        super(Dataset, self).__init__()
        self.is_train = is_train
        self.SNR = SNR
        self.ir = ir

        img_data = hdf5storage.loadmat(os.path.join(file_path, 'imgs_test.mat'))['imgs']  # B, 3, 384, 384
        gps_data = hdf5storage.loadmat(os.path.join(file_path, 'gps_test.mat'))['gps']  # B, 3
        csi_data = hdf5storage.loadmat(os.path.join(file_path, 'H_test.mat'))['H']  # B, N, K

        csi_data = noise(csi_data, SNR)

        print(img_data.shape, gps_data.shape, csi_data.shape)
        # Load img
        self.imgs = torch.tensor(img_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        # Load data for Channel Estimation(T1)
        self.T1_in = csi_data[:, :, ::num_pilot]  # 1500, 16, 8
        self.T1_out = csi_data  # 1500, 16, 64
        self.T1_in = LoadBatch(self.T1_in)
        self.T1_out = LoadBatch(self.T1_out)
        # Load data for Channel Prediction(T2)
        self.T2_in = csi_data[:, :, :prev_len]  # 1500, 16, 32
        self.T2_out = csi_data[:, :, prev_len:]  # 1500, 16, 32
        self.T2_in = LoadBatch(self.T2_in)
        self.T2_out = LoadBatch(self.T2_out)
        # Load data for Vision-aided Position (T3)
        self.T3_in = csi_data[:, :, :]  # 1500, 16, 64
        self.T3_out = torch.tensor(gps_data[:, :2], dtype=torch.float32)  # 1500, 2
        self.T3_in = LoadBatch(self.T3_in)
        self.T3_out = (self.T3_out - self.T3_out.mean()) / self.T3_out.std()
        self.T3_out = self.T3_out


        if is_show:
            print('Training Dataset info: ')
            print(
                f'Task1 in shape: {self.T1_in.shape}\t'
                f'Task1 out shape: {self.T1_out.shape}\n'
                f'Task2 in shape: {self.T2_in.shape}\t'
                f'Task2 out shape: {self.T2_out.shape}\n'
                f'Task3 rgb shape: {self.T3_in.shape}\t'
                f'Task3 out shape: {self.T3_out.shape}\n'
                f'img shape: {self.imgs.shape}\n'
            )

    def __getitem__(self, index):
        return {
            "T1i": self.T1_in[index, ...].float(),
            "T1o": self.T1_out[index, ...].float(),
            "T2i": self.T2_in[index, ...].float(),
            "T2o": self.T2_out[index, ...].float(),
            "T3i": self.T3_in[index, ...].float(),
            "T3o": self.T3_out[index, ...].float(),
            "img": self.imgs[index],
        }

    def __len__(self):
        return self.T1_in.shape[0]


if __name__ == '__main__':
    batch_size = 32
    path1 = '/data1/PCNI1_data/MTLLM_llama/src/MLoRA_SoM_open_source/data/M3C'
    data_set = Dataset(path1, SNR=10, is_train=0, valid_per=0.1)


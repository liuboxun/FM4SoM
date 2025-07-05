# coding=utf-8
import numpy as np
import torch as th
import os
import json
import torch
import scipy.io
import datetime
import copy
import hdf5storage
import random
import math
from torch.utils.data import Dataset
import torch.nn.functional as F
from einops import rearrange

def LoadBatch(H):
    # H: ...     [tensor complex]
    # out: ..., 2  [tensor real]
    size = list(H.shape)
    H_real = np.zeros(size + [2])
    H_real[..., 0] = H.real
    H_real[..., 1] = H.imag
    H_real = torch.tensor(H_real, dtype=torch.float32)
    H_real = rearrange(H_real, 'b n k o -> b o n k')
    return H_real


class Dataset_vision_aided_cpf(Dataset):
    def __init__(self, file_path, is_train=1, ir=1, is_show=1):
        super(Dataset_vision_aided_cpf, self).__init__()
        self.is_train = is_train
        self.ir = ir
        # Shuffle and Segmentation Dateset
        db = hdf5storage.loadmat(file_path+'/H_test.mat')['H']
        db2 = hdf5storage.loadmat(file_path+'/imgs_test.mat')['imgs']
        # Load data for Channel Prediction
        self.H = torch.tensor(db, dtype=torch.complex128)
        self.img = torch.tensor(db2, dtype=torch.float32).permute(0, 3, 1, 2)

        self.H = LoadBatch(self.H)
        self.img = F.interpolate(self.img, size=(224, 224), mode='bilinear', align_corners=False) / 255.0

        if is_show:
            print('Training Dataset info: ')
            print(
                f'image shape: {self.img.shape}\t'
                f'channel shape: {self.H.shape}\n'
            )

    def __getitem__(self, index):
        return {
            "img": self.img[index].float(),
            "h": self.H[index, ...].float(),
        }

    def __len__(self):
        return self.H.shape[0]

def data_load_vision_aided_cpf(args):
    test_data = Dataset_vision_aided_cpf(args.file_load_path, is_train=0)
    test_data = th.utils.data.DataLoader(test_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=False, prefetch_factor=4)

    return test_data


if __name__ == '__main__':
    path = '../dataset/M3C'
    dataset = Dataset_vision_aided_cpf(path)
    # for key, value in dataset[0].items():
    #     print(key, value.shape)
    #     print(value)

import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from einops import rearrange
from numpy import random


def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path_in, file_path_out, is_train=1,
                 train_per=0.9, valid_per=0.1):
        super(Dataset_Pro, self).__init__()

#        读取数据
        H_in = hdf5storage.loadmat(file_path_in)['LiDAR_point']  # v,b,l,k,a,b,c
        H_out = hdf5storage.loadmat(file_path_out)["scatterer_num"]  # v,b,l,k,a,b,c
        print(H_in.shape, H_out.shape)

        batch = H_out.shape[0]
        if is_train:
            H_in = H_in[:int(train_per * batch), ...]
            H_out = H_out[:int(train_per * batch), ...]
        else:
            H_in = H_in[int(train_per * batch):int((train_per + valid_per) * batch), ...]
            H_out = H_out[int(train_per * batch):int((train_per + valid_per) * batch), ...]
        print('H_out.shape[0]:', H_out.shape[0])
#        H_in = rearrange(H_in, 'v n L k a b c -> (v n) L (k a b c)')
#        H_out = rearrange(H_out, 'v n L k a b c -> (v n) L (k a b c)')

#        B, prev_len, mul = H_in.shape
#        _, pred_len, mul = H_out.shape
#        self.pred_len = pred_len
#        self.prev_len = prev_len
#        self.seq_len = pred_len + prev_len
#
#        dt_all = np.concatenate((H_in, H_out), axis=1)
#        np.random.shuffle(dt_all)
#        H_in = dt_all[:, :prev_len, ...]
#        H_out = dt_all[:, -pred_len:, ...]
#        std = np.sqrt(np.std(np.abs(H_in) ** 2))
#        H_in = H_in / std
#        H_out = H_out / std
        self.pred = H_out  # b,16,(48*2)
        self.prev = H_in  # b,4,(48*2)

    def __getitem__(self, index):
        return self.pred[index, :], \
               self.prev[index, :]
#        return self.pred[index, :].float(), \
#               self.prev[index, :].float()

    def __len__(self):
        return self.pred.shape[0]



class Dataset_Pro2(data.Dataset):
    def __init__(self, file_path_in, file_path_out, is_train=1,
                 train_per=0.9, valid_per=0.1):
        super(Dataset_Pro2, self).__init__()

#        读取数据
        H_in = hdf5storage.loadmat(file_path_in)['LiDAR_point3']  # v,b,l,k,a,b,c
        H_out = hdf5storage.loadmat(file_path_out)["scatterer_gridmap2"]  # v,b,l,k,a,b,c
        print(H_in.shape, H_out.shape)

        batch = H_out.shape[0]
        if is_train:
            H_in = H_in[:int(train_per * batch), ...]
            H_out = H_out[:int(train_per * batch), ...]
        else:
            H_in = H_in[int(train_per * batch):int((train_per + valid_per) * batch), ...]
            H_out = H_out[int(train_per * batch):int((train_per + valid_per) * batch), ...]
        print('H_out.shape[0]:', H_out.shape[0])
#        H_in = rearrange(H_in, 'v n L k a b c -> (v n) L (k a b c)')
#        H_out = rearrange(H_out, 'v n L k a b c -> (v n) L (k a b c)')

#        B, prev_len, mul = H_in.shape
#        _, pred_len, mul = H_out.shape
#        self.pred_len = pred_len
#        self.prev_len = prev_len
#        self.seq_len = pred_len + prev_len
#
#        dt_all = np.concatenate((H_in, H_out), axis=1)
#        np.random.shuffle(dt_all)
#        H_in = dt_all[:, :prev_len, ...]
#        H_out = dt_all[:, -pred_len:, ...]
#        std = np.sqrt(np.std(np.abs(H_in) ** 2))
#        H_in = H_in / std
#        H_out = H_out / std
        self.pred = H_out  # b,16,(48*2)
        self.prev = H_in  # b,4,(48*2)

    def __getitem__(self, index):
        return self.pred[index, :], \
               self.prev[index, :]
#        return self.pred[index, :].float(), \
#               self.prev[index, :].float()

    def __len__(self):
        return self.pred.shape[0]


class Dataset_Pro3(data.Dataset):
    def __init__(self, file_path_in, file_path_out, is_train=1,
                 train_per=0.9, valid_per=0.1):
        super(Dataset_Pro3, self).__init__()

#        读取数据
        H_in1 = hdf5storage.loadmat(file_path_in)['LiDAR_point1']  # v,b,l,k,a,b,c
        H_out = hdf5storage.loadmat(file_path_out)["scatterer_gridmap"]  # v,b,l,k,a,b,c
        print(H_in1.shape, H_in2.shape, H_out.shape)
        H_in=np.concatenate((H_in1, H_in2), axis=0)
        print(H_in.shape)

        batch = H_out.shape[0]
        if is_train:
            H_in = H_in[:int(train_per * batch), ...]
            H_out = H_out[:int(train_per * batch), ...]
        else:
            H_in = H_in[int(train_per * batch):int((train_per + valid_per) * batch), ...]
            H_out = H_out[int(train_per * batch):int((train_per + valid_per) * batch), ...]
        print('H_out.shape[0]:', H_out.shape[0])
#        H_in = rearrange(H_in, 'v n L k a b c -> (v n) L (k a b c)')
#        H_out = rearrange(H_out, 'v n L k a b c -> (v n) L (k a b c)')

#        B, prev_len, mul = H_in.shape
#        _, pred_len, mul = H_out.shape
#        self.pred_len = pred_len
#        self.prev_len = prev_len
#        self.seq_len = pred_len + prev_len
#
#        dt_all = np.concatenate((H_in, H_out), axis=1)
#        np.random.shuffle(dt_all)
#        H_in = dt_all[:, :prev_len, ...]
#        H_out = dt_all[:, -pred_len:, ...]
#        std = np.sqrt(np.std(np.abs(H_in) ** 2))
#        H_in = H_in / std
#        H_out = H_out / std
        self.pred = H_out  # b,16,(48*2)
        self.prev = H_in  # b,4,(48*2)

    def __getitem__(self, index):
        return self.pred[index, :], \
               self.prev[index, :]
#        return self.pred[index, :].float(), \
#               self.prev[index, :].float()

    def __len__(self):
        return self.pred.shape[0]

class Dataset_Pro4(data.Dataset):
    def __init__(self, file_path_in, file_path_in2, file_path_out, is_train=1,
                 train_per=0.9, valid_per=0.1):
        super(Dataset_Pro4, self).__init__()

#        读取数据
        H_in1 = hdf5storage.loadmat(file_path_in)['LiDAR_point1']  # v,b,l,k,a,b,c
        H_in2 = hdf5storage.loadmat(file_path_in2)['LiDAR_point2']  # v,b,l,k,a,b,c
        H_out = hdf5storage.loadmat(file_path_out)["scatterer_gridmap1"]  # v,b,l,k,a,b,c
        print(H_in1.shape, H_in2.shape, H_out.shape)
        H_in=np.concatenate((H_in1, H_in2), axis=0)
        print(H_in.shape)

        batch = H_out.shape[0]
        if is_train:
            H_in = H_in[:int(train_per * batch), ...]
            H_out = H_out[:int(train_per * batch), ...]
        else:
            H_in = H_in[int(train_per * batch):int((train_per + valid_per) * batch), ...]
            H_out = H_out[int(train_per * batch):int((train_per + valid_per) * batch), ...]
        print('H_out.shape[0]:', H_out.shape[0])
#        H_in = rearrange(H_in, 'v n L k a b c -> (v n) L (k a b c)')
#        H_out = rearrange(H_out, 'v n L k a b c -> (v n) L (k a b c)')

#        B, prev_len, mul = H_in.shape
#        _, pred_len, mul = H_out.shape
#        self.pred_len = pred_len
#        self.prev_len = prev_len
#        self.seq_len = pred_len + prev_len
#
#        dt_all = np.concatenate((H_in, H_out), axis=1)
#        np.random.shuffle(dt_all)
#        H_in = dt_all[:, :prev_len, ...]
#        H_out = dt_all[:, -pred_len:, ...]
#        std = np.sqrt(np.std(np.abs(H_in) ** 2))
#        H_in = H_in / std
#        H_out = H_out / std
        self.pred = H_out  # b,16,(48*2)
        self.prev = H_in  # b,4,(48*2)

    def __getitem__(self, index):
        return self.pred[index, :], \
               self.prev[index, :]
#        return self.pred[index, :].float(), \
#               self.prev[index, :].float()

    def __len__(self):
        return self.pred.shape[0]


        


def LoadBatch_ofdm_2(H):
    # H: B,T,K,mul     [tensor complex]
    # out:B,T,K,mul*2  [tensor real]
    B, T, K, mul = H.shape
    H_real = np.zeros([B, T, K, mul, 2])
    H_real[:, :, :, :, 0] = H.real
    H_real[:, :, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, K, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm_1(H):
    # H: B,T,mul     [tensor complex]
    # out:B,T,mul*2  [tensor real]
    B, T, mul = H.shape
    H_real = np.zeros([B, T, mul, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm(H, num=32):
    # H: B,T,mul             [tensor complex]
    # out:B*num,T,mul*2/num  [tensor real]
    B, T, mul = H.shape
    H = rearrange(H, 'b t (k a) ->(b a) t k', a=num)
    H_real = np.zeros([B * num, T, mul // num, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B * num, T, mul // num * 2])
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




def create_samples(path_in, path_out):
#    init_names_in = os.listdir(path_in) # List all sub-directories in in_path
#    init_names_out = os.listdir(path_out)
#    if nat_sort:
#        sub_dir_names_in = natsorted(init_names_in) # sort directory names in natural order
#                                              # (Only for directories with numbers for names)
#        sub_dir_names_out = natsorted(init_names_out)
#    else:
#        sub_dir_names_in = init_names_in
#        sub_dir_names_out = init_names_out
    data_samples = []
#    for sub_dir in sub_dir_names: # Loop over all sub-directories
#        per_dir = os.listdir(root+'/'+sub_dir) # Get a list of names from sub-dir # i

    RGB_list = []
    pl_list = []
    image_num = 0
    for name in path_in:
        image_num = image_num + 1
#        split_name = name.split('_')
        
        RGB_list = (path_in + '/' + 'image' + '/' + str(image_num))
        pl_list = (path_out + '/' + 'image' + '/' + str(image_num))
        sample = (RGB_list,pl_list)
        data_samples.append(sample)
    

    return data_samples
    
class Dataset_Pro5(data.Dataset):
    def __init__(self, file_path_in, file_path_out):
        super(Dataset_Pro5, self).__init__()
        self.path_in = file_path_in
        self.path_out = file_path_out
        self.samples = create_samples(self.path_in, self.path_out) 

	   
    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        RGB_list = sample[0]
        for i in RGB_list:
            split_i = i.split('/')
            split_i = split_i[2]
            split_i = split_i.split('.')
            split_i = split_i[0]
            RGB = sciio.loadmat(i)[split_i]
            RGB = torch.as_tensor(building)
        label_list = sample[1]
        for j in pl_list:
            split_j = j.split('/')
            split_j = split_j[2]
            split_j = split_j.split('.')
            split_j = split_j[0]
            pl = sciio.loadmat(j)[split_j]
            pl = torch.as_tensor(pl)
        return (RGB,pl)

class Dataset_Pro6(data.Dataset):
    def __init__(self, file_path_in, file_path_out, is_train=1):
        super(Dataset_Pro6, self).__init__()

#        读取数据
        H_in1 = hdf5storage.loadmat(file_path_in)['LiDAR_point1']  # v,b,l,k,a,b,c
        H_in2 = hdf5storage.loadmat(file_path_in2)['LiDAR_point2']  # v,b,l,k,a,b,c
        H_out = hdf5storage.loadmat(file_path_out)["scatterer_gridmap1"]  # v,b,l,k,a,b,c
        print(H_in1.shape, H_in2.shape, H_out.shape)
        H_in=np.concatenate((H_in1, H_in2), axis=0)
        print(H_in.shape)

        batch = H_out.shape[0]
        if is_train:
            H_in = H_in[:int(train_per * batch), ...]
            H_out = H_out[:int(train_per * batch), ...]
        else:
            H_in = H_in[int(train_per * batch):int((train_per + valid_per) * batch), ...]
            H_out = H_out[int(train_per * batch):int((train_per + valid_per) * batch), ...]
        print('H_out.shape[0]:', H_out.shape[0])
#        H_in = rearrange(H_in, 'v n L k a b c -> (v n) L (k a b c)')
#        H_out = rearrange(H_out, 'v n L k a b c -> (v n) L (k a b c)')

#        B, prev_len, mul = H_in.shape
#        _, pred_len, mul = H_out.shape
#        self.pred_len = pred_len
#        self.prev_len = prev_len
#        self.seq_len = pred_len + prev_len
#
#        dt_all = np.concatenate((H_in, H_out), axis=1)
#        np.random.shuffle(dt_all)
#        H_in = dt_all[:, :prev_len, ...]
#        H_out = dt_all[:, -pred_len:, ...]
#        std = np.sqrt(np.std(np.abs(H_in) ** 2))
#        H_in = H_in / std
#        H_out = H_out / std
        self.pred = H_out  # b,16,(48*2)
        self.prev = H_in  # b,4,(48*2)

    def __getitem__(self, index):
        return self.pred[index, :], \
               self.prev[index, :]
#        return self.pred[index, :].float(), \
#               self.prev[index, :].float()

    def __len__(self):
        return self.pred.shape[0]



import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class Dataset_Pro7(Dataset):
    def __init__(self, rgb_dir, pl_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.pl_dir = pl_dir
        self.rgb_images = sorted(os.listdir(rgb_dir))
        self.pl_images = sorted(os.listdir(pl_dir))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        rgb_image = Image.open(rgb_path).convert('RGB')  # 确保为RGB格式
        pl_image = Image.open(pl_path).convert('RGB')    # 如果PL图像也是RGB
        
        rgb_image = self.transform(rgb_image)
        pl_image = self.transform(pl_image)
        
        return rgb_image, pl_image



class Dataset_Pro8(Dataset):
    def __init__(self, dep_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.pl_dir = pl_dir
        self.dep_images = sorted(os.listdir(dep_dir))
        self.pl_images = sorted(os.listdir(pl_dir))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('L')  # 确保为RGB格式
        pl_image = Image.open(pl_path).convert('L')    # 如果PL图像也是RGB
        
        dep_image = self.transform(dep_image)
        pl_image = self.transform(pl_image)
        
        return dep_image, pl_image



class Dataset_Pro9(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        self.dep_images = sorted(os.listdir(dep_dir))
        self.RGB_images = sorted(os.listdir(RGB_dir))
        self.pl_images = sorted(os.listdir(pl_dir))
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),            # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    

class Dataset_Pro_dep(Dataset):
    def __init__(self, dep_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.pl_dir = pl_dir
        self.dep_images = sorted(os.listdir(dep_dir))
        self.pl_images = sorted(os.listdir(pl_dir))
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, pl_image
    

class Dataset_Pro_rgb(Dataset):
    def __init__(self, RGB_dir, pl_dir, transform=None):
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        self.RGB_images = sorted(os.listdir(RGB_dir))
        self.pl_images = sorted(os.listdir(pl_dir))
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.RGB_images)

    def __getitem__(self, idx):
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return RGB_image, pl_image

class Dataset_Pro99(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        self.dep_images = sorted(os.listdir(dep_dir))
        self.RGB_images = sorted(os.listdir(RGB_dir))
        self.pl_images = sorted(os.listdir(pl_dir))
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image


class Dataset_Pro10(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        self.dep_images = sorted(os.listdir(dep_dir))
        self.RGB_images = sorted(os.listdir(RGB_dir))
        self.pl_images = sorted(os.listdir(pl_dir))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((256, 256)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image



class Dataset_Pro_ck_28(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        self.dep_images = sorted(os.listdir(dep_dir))
        self.RGB_images = sorted(os.listdir(RGB_dir))
        self.pl_images = sorted(os.listdir(pl_dir))
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((32, 32)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    
class Dataset_Pro11(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        all_dep_images = sorted(os.listdir(dep_dir))
        all_RGB_images = sorted(os.listdir(RGB_dir))
        all_pl_images = sorted(os.listdir(pl_dir))
        # 随机取1000个样本，如果总数不足1000则取全部
        if len(all_dep_images) > 1000:
            indices = np.random.choice(len(all_dep_images), 1000, replace=False)
            self.dep_images = [all_dep_images[i] for i in indices]
            self.RGB_images = [all_RGB_images[i] for i in indices]
            self.pl_images = [all_pl_images[i] for i in indices]
        else:
            self.dep_images = all_dep_images
            self.RGB_images = all_RGB_images
            self.pl_images = all_pl_images
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    


class Dataset_Pro_500(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        all_dep_images = sorted(os.listdir(dep_dir))
        all_RGB_images = sorted(os.listdir(RGB_dir))
        all_pl_images = sorted(os.listdir(pl_dir))
        # 随机取1000个样本，如果总数不足1000则取全部
        if len(all_dep_images) > 500:
            indices = np.random.choice(len(all_dep_images), 500, replace=False)
            self.dep_images = [all_dep_images[i] for i in indices]
            self.RGB_images = [all_RGB_images[i] for i in indices]
            self.pl_images = [all_pl_images[i] for i in indices]
        else:
            self.dep_images = all_dep_images
            self.RGB_images = all_RGB_images
            self.pl_images = all_pl_images
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    
class Dataset_Pro_400(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        all_dep_images = sorted(os.listdir(dep_dir))
        all_RGB_images = sorted(os.listdir(RGB_dir))
        all_pl_images = sorted(os.listdir(pl_dir))
        # 随机取1000个样本，如果总数不足1000则取全部
        if len(all_dep_images) > 400:
            indices = np.random.choice(len(all_dep_images), 400, replace=False)
            self.dep_images = [all_dep_images[i] for i in indices]
            self.RGB_images = [all_RGB_images[i] for i in indices]
            self.pl_images = [all_pl_images[i] for i in indices]
        else:
            self.dep_images = all_dep_images
            self.RGB_images = all_RGB_images
            self.pl_images = all_pl_images
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    

class Dataset_Pro_300(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        all_dep_images = sorted(os.listdir(dep_dir))
        all_RGB_images = sorted(os.listdir(RGB_dir))
        all_pl_images = sorted(os.listdir(pl_dir))
        # 随机取1000个样本，如果总数不足1000则取全部
        if len(all_dep_images) > 300:
            indices = np.random.choice(len(all_dep_images), 300, replace=False)
            self.dep_images = [all_dep_images[i] for i in indices]
            self.RGB_images = [all_RGB_images[i] for i in indices]
            self.pl_images = [all_pl_images[i] for i in indices]
        else:
            self.dep_images = all_dep_images
            self.RGB_images = all_RGB_images
            self.pl_images = all_pl_images
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    

class Dataset_Pro_200(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        all_dep_images = sorted(os.listdir(dep_dir))
        all_RGB_images = sorted(os.listdir(RGB_dir))
        all_pl_images = sorted(os.listdir(pl_dir))
        # 随机取1000个样本，如果总数不足1000则取全部
        if len(all_dep_images) > 200:
            indices = np.random.choice(len(all_dep_images), 200, replace=False)
            self.dep_images = [all_dep_images[i] for i in indices]
            self.RGB_images = [all_RGB_images[i] for i in indices]
            self.pl_images = [all_pl_images[i] for i in indices]
        else:
            self.dep_images = all_dep_images
            self.RGB_images = all_RGB_images
            self.pl_images = all_pl_images
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
        
class Dataset_Pro_100(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        all_dep_images = sorted(os.listdir(dep_dir))
        all_RGB_images = sorted(os.listdir(RGB_dir))
        all_pl_images = sorted(os.listdir(pl_dir))
        # 随机取1000个样本，如果总数不足1000则取全部
        if len(all_dep_images) > 100:
            indices = np.random.choice(len(all_dep_images), 100, replace=False)
            self.dep_images = [all_dep_images[i] for i in indices]
            self.RGB_images = [all_RGB_images[i] for i in indices]
            self.pl_images = [all_pl_images[i] for i in indices]
        else:
            self.dep_images = all_dep_images
            self.RGB_images = all_RGB_images
            self.pl_images = all_pl_images
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    

class Dataset_Pro_50(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        all_dep_images = sorted(os.listdir(dep_dir))
        all_RGB_images = sorted(os.listdir(RGB_dir))
        all_pl_images = sorted(os.listdir(pl_dir))
        # 随机取1000个样本，如果总数不足1000则取全部
        if len(all_dep_images) > 50:
            indices = np.random.choice(len(all_dep_images), 50, replace=False)
            self.dep_images = [all_dep_images[i] for i in indices]
            self.RGB_images = [all_RGB_images[i] for i in indices]
            self.pl_images = [all_pl_images[i] for i in indices]
        else:
            self.dep_images = all_dep_images
            self.RGB_images = all_RGB_images
            self.pl_images = all_pl_images
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    

class Dataset_Pro_40(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        all_dep_images = sorted(os.listdir(dep_dir))
        all_RGB_images = sorted(os.listdir(RGB_dir))
        all_pl_images = sorted(os.listdir(pl_dir))
        # 随机取1000个样本，如果总数不足1000则取全部
        if len(all_dep_images) > 40:
            indices = np.random.choice(len(all_dep_images), 40, replace=False)
            self.dep_images = [all_dep_images[i] for i in indices]
            self.RGB_images = [all_RGB_images[i] for i in indices]
            self.pl_images = [all_pl_images[i] for i in indices]
        else:
            self.dep_images = all_dep_images
            self.RGB_images = all_RGB_images
            self.pl_images = all_pl_images
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    
class Dataset_Pro_30(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        all_dep_images = sorted(os.listdir(dep_dir))
        all_RGB_images = sorted(os.listdir(RGB_dir))
        all_pl_images = sorted(os.listdir(pl_dir))
        # 随机取1000个样本，如果总数不足1000则取全部
        if len(all_dep_images) > 30:
            indices = np.random.choice(len(all_dep_images), 30, replace=False)
            self.dep_images = [all_dep_images[i] for i in indices]
            self.RGB_images = [all_RGB_images[i] for i in indices]
            self.pl_images = [all_pl_images[i] for i in indices]
        else:
            self.dep_images = all_dep_images
            self.RGB_images = all_RGB_images
            self.pl_images = all_pl_images
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  # 确保为RGB格式
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
##        读取数据
#        H_in1 = hdf5storage.loadmat(file_path_in)['LiDAR_point1']  # v,b,l,k,a,b,c
#        H_out = hdf5storage.loadmat(file_path_out)["scatterer_gridmap"]  # v,b,l,k,a,b,c
#        print(H_in1.shape, H_in2.shape, H_out.shape)
#        H_in=np.concatenate((H_in1, H_in2), axis=0)
#        print(H_in.shape)
#
#        batch = H_out.shape[0]
#        if is_train:
#            H_in = H_in[:int(train_per * batch), ...]
#            H_out = H_out[:int(train_per * batch), ...]
#        else:
#            H_in = H_in[int(train_per * batch):int((train_per + valid_per) * batch), ...]
#            H_out = H_out[int(train_per * batch):int((train_per + valid_per) * batch), ...]
#        print('H_out.shape[0]:', H_out.shape[0])
##        H_in = rearrange(H_in, 'v n L k a b c -> (v n) L (k a b c)')
##        H_out = rearrange(H_out, 'v n L k a b c -> (v n) L (k a b c)')
#
##        B, prev_len, mul = H_in.shape
##        _, pred_len, mul = H_out.shape
##        self.pred_len = pred_len
##        self.prev_len = prev_len
##        self.seq_len = pred_len + prev_len
##
##        dt_all = np.concatenate((H_in, H_out), axis=1)
##        np.random.shuffle(dt_all)
##        H_in = dt_all[:, :prev_len, ...]
##        H_out = dt_all[:, -pred_len:, ...]
##        std = np.sqrt(np.std(np.abs(H_in) ** 2))
##        H_in = H_in / std
##        H_out = H_out / std
#        self.pred = H_out  # b,16,(48*2)
#        self.prev = H_in  # b,4,(48*2)

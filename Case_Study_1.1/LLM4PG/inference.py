import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datapro.data_provider import Dataset_Pro9
import scipy.io as sio
# from models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction,PointNetFeaturePropagation
from models.GPT2_rgb_d_patching import GPTRGBD, Conv_patching_RGB, Conv_patching_dep
import numpy as np
import shutil
import argparse
from torch.utils.tensorboard import SummaryWriter
import torchvision
import imageio
from tqdm import tqdm
from torchvision import transforms
import time

# from metrics import NMSELoss, SE_Loss


batch_size = 1
# device = torch.device('cuda:1')
# torch.cuda.set_device(2)
device = torch.device('cuda:3')

load_path_rgb = "weights/wt/pl_patching_0522_rgbd_rgb_bs128_epoch201_lr0.0001-70-28-full.pth"
load_path_dep = "weights/wt/pl_patching_0522_rgbd_dep_bs128_epoch201_lr0.0001-70-28-full.pth"
load_path_gpt = "weights/wt/pl_patching_0522_rgbd_gpt_bs128_epoch201_lr0.0001-70-28-full.pth"



test_dep_path = "xx/dep/"
test_RGB_path = "xx/RGB/"
test_pl_path = "xx/pl/"

test_set = Dataset_Pro9(test_dep_path, test_RGB_path, test_pl_path, transform=None)  # creat data for training

parser = argparse.ArgumentParser(description='GPTRGBD')
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--stride', type=int, default=64)
parser.add_argument('--seq_len', type=int, default=1024)

parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--freeze_inout', type=int, default=0)
parser.add_argument('--freeze_gptall', type=int, default=0)
parser.add_argument('--freeze_pointnet', type=int, default=0)

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--pred_len', type=int, default=100)

args = parser.parse_args()
model_GPT = GPTRGBD(args, device).to(device)
model_CNN_RGB = Conv_patching_RGB(args, device).to(device)
model_CNN_dep = Conv_patching_dep(args, device).to(device)


model_CNN_dep = torch.load(load_path_dep, map_location=device)
model_CNN_RGB = torch.load(load_path_rgb, map_location=device)
model_GPT = torch.load(load_path_gpt, map_location=device)

# 加载参数
if os.path.exists(load_path_dep):
    model_CNN_dep = torch.load(load_path_dep, map_location=device)
if os.path.exists(load_path_rgb):
    model_CNN_RGB = torch.load(load_path_rgb, map_location=device)
if os.path.exists(load_path_gpt):
    model_GPT = torch.load(load_path_gpt, map_location=device)


###################################################################
# ------------------- Main test (Run second)----------------------------------
###################################################################
def test(testing_data_loader):
    global total_loss
    print('Start testing...!!!!!!!!!!!')
    epoch_test_loss, epoch_test_loss2 = [], []
    progress_bar = tqdm(enumerate(testing_data_loader), total=len(testing_data_loader), desc=f'Testing', leave=True)

    model_CNN_dep.eval()
    model_CNN_RGB.eval()
    model_GPT.eval()
    with torch.no_grad():
        i = 0
        all_nmse = []
        for iteration, batch in enumerate(testing_data_loader, 1):
            
            
            prev_dep, prev_RGB, pred_t = Variable(batch[0]).to(device).float(), \
                    Variable(batch[1]).to(device).float(),\
                    Variable(batch[2]).to(device).float()
                #                optimizer.zero_grad()  # fixed
            start = time.time()
            dep_features = model_CNN_dep(prev_dep)
            rgb_features = model_CNN_RGB(prev_RGB)

            rgb_d_features = torch.cat((dep_features, rgb_features), dim = 2)

            
            pred_m = model_GPT(rgb_d_features)
            
            end = time.time()
            test_time = end - start
            print('test time: {:.7f}'.format(test_time))
            #print('pred_m',pred_m.shape)
            #print('pred_t',pred_t.shape)   
            #print("RGB",prev_RGB.shape) 
            #trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            #pl_image_3c = trans_norm(pl_pre)
            # pl_image_3c = pl_pre.repeat(1, 3, 1, 1)
            # pred_m : prediction
            pred_m_3c = pred_m.repeat(1, 3, 1, 1)
            pred_t_3c = pred_t.repeat(1, 3, 1, 1)
            pl_va = (pred_t - pred_t.min()) * (255 / (pred_t.max() - pred_t.min()))
            pl_va = pl_va.cpu()
            pl_pre_va = (pred_m - pred_m.min()) * (255 / (pred_m.max() - pred_m.min()))
            pl_pre_va = pl_pre_va.cpu()
            #print('pl_va',pl_va)
            #print('pl_pre_va',pl_pre_va)
            #print('pl_va-pl_pre_va',pl_va - pl_pre_va)
            error = torch.sum((pl_va - pl_pre_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            all_nmse.append(nmse)
            all_nmse_mean = np.mean(all_nmse)
            print("i",i)
            print(f"NMSE: {nmse}")
            print(f"NMSE_mean: {all_nmse_mean}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            prev_RGB,
                            prev_dep,
                            pred_m_3c,
                            pred_t_3c,
                           
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(
                os.path.join('/home/smr/LLM4PP/experiments', f"0716/wt_test_patching_{i}_epoch201-63-28-full.jpg"),
                merged)
            i = i + 1
            # print('pred_t',pred_t)
            # print('pred_m',pred_m)
            # loss = criterion(pred_m, pred_t)  # compute loss
            # epoch_test_loss.append(loss.item())  # save all losses into a vector for one epoch
            # loss2 = criterion2(pred_m, pred_t)  # compute loss
            #           print(loss2)
            # epoch_test_loss2.append(loss2.item())  # save all losses into a vector for one epoch
            #           print(len(epoch_test_loss2))
            progress_bar.set_postfix(loss=nmse.item())
            progress_bar.update()  # 更新进度条
        # v_loss = np.nanmean(np.array(epoch_test_loss))
        # v_loss2 = np.nanmean(np.array(epoch_test_loss2))
        # print('validate loss: {:.7f}'.format(v_loss))
        # print('validate loss2: {:.7f}'.format(v_loss2))


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.5fM" % (total / 1e6))
    # total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                     pin_memory=True,
                                     drop_last=True)  # put testing data to DataLoader for batches

    #    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
    #    criterion = NMSELoss().to(device)
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    test(testing_data_loader)  # call test function (

    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.5fM" % (total / 1e6))
    # total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

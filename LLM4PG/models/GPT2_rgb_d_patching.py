import numpy as np
import torch
import torch.nn as nn
from torch import optim
import argparse

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import torch.nn as nn
import torch.nn.functional as F
import torch


class Conv_patching_RGB(nn.Module):
    def __init__(self, configs, device, normal_channel=False):
        super(Conv_patching_RGB, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        

        num_patches = 64
        embed_dim = 384
        self.conv_patch = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=8, stride=8) #RGB 256*256
        self.bn_patch = nn.BatchNorm2d(embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # [1, max_num_patches, 768]

    def forward(self, x):
        #print('00:',x.shape)
        x = self.conv_patch(x)
        #print('001:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = self.bn_patch(x)
        #print('002:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = nn.functional.relu(x)
        #print('003:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        outputs = x.flatten(2).transpose(1, 2)
        #print('004:',outputs.shape)
        
        outputs = outputs + self.position_encoding
        #print('006:',outputs.shape) # [batch_size, 64]
        return outputs


class Conv_patching_RGB_2(nn.Module):
    def __init__(self, configs, device, normal_channel=False):
        super(Conv_patching_RGB_2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        

        num_patches = 64
        embed_dim = 768
        self.conv_patch = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=8, stride=8) #RGB 256*256
        self.bn_patch = nn.BatchNorm2d(embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # [1, max_num_patches, 768]

    def forward(self, x):
        #print('00:',x.shape)
        x = self.conv_patch(x)
        #print('001:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = self.bn_patch(x)
        #print('002:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = nn.functional.relu(x)
        #print('003:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        outputs = x.flatten(2).transpose(1, 2)
        #print('004:',outputs.shape)
        
        outputs = outputs + self.position_encoding
        #print('006:',outputs.shape) # [batch_size, 64]
        return outputs

class Conv_patching_dep(nn.Module):
    def __init__(self, configs, device, normal_channel=False):
        super(Conv_patching_dep, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        

        num_patches = 64
        embed_dim = 384
        self.conv_patch = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=8, stride=8) #RGB 256*256
        self.bn_patch = nn.BatchNorm2d(embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # [1, max_num_patches, 768]

    def forward(self, x):
        #print('00:',x.shape)
        x = self.conv_patch(x)
        #print('01:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = self.bn_patch(x)
        #print('02:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = nn.functional.relu(x)
        #print('03:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true

        outputs = x.flatten(2).transpose(1, 2)
        #print('04:',outputs.shape)
        outputs = outputs + self.position_encoding

        #print('06:',outputs.shape) # [batch_size, 64]
        
        return outputs

class Conv_patching_dep_2(nn.Module):
    def __init__(self, configs, device, normal_channel=False):
        super(Conv_patching_dep_2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        

        num_patches = 64
        embed_dim = 768
        self.conv_patch = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=8, stride=8) #RGB 256*256
        self.bn_patch = nn.BatchNorm2d(embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # [1, max_num_patches, 768]

    def forward(self, x):
        #print('00:',x.shape)
        x = self.conv_patch(x)
        #print('01:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = self.bn_patch(x)
        #print('02:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = nn.functional.relu(x)
        #print('03:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true

        outputs = x.flatten(2).transpose(1, 2)
        #print('04:',outputs.shape)
        outputs = outputs + self.position_encoding

        #print('06:',outputs.shape) # [batch_size, 64]
        
        return outputs

class GPTRGBD(nn.Module): 
    def __init__(self, configs, device,normal_channel=False):
        super(GPTRGBD, self).__init__()
    
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
#        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(768, 65536)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # (B, 256, 16, 16) -> (B, 128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            #(B, 128, 32, 32) -> (B, 64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # # (B, 64, 64, 64) -> (B, 32, 128, 128)
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2),
            # # (B, 32, 64, 64) -> (B, 16, 256, 256)
            # nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(0.2),

            # (B, 16, 64, 64) -> (B, 3, 64, 64)
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        
#        self.out_layer = nn.Linear(configs.d_model * self.patch_num, 196608)

#        冻结部分参数
#        if configs.freeze_gptall and configs.pretrain:
#            print('configs.freeze_gptall')
#            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#                param.requires_grad = False
#        elif configs.freeze and configs.pretrain:
#            print('configs.freeze')
#            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#                if 'ln' in name or 'wpe' in name:
#                    param.requires_grad = True
#                else:
#                    param.requires_grad = False

#        各层转移到device上
        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


#    def forward(self, x, itr):
    def forward(self, x):
#        B, _, _ = xyz.shape
#        if self.normal_channel:
#            norm = xyz[:, 3:, :]
#            xyz = xyz[:, :3, :]
#        else:
#            norm = None
#        l1_xyz, l1_points = self.sa1(xyz, norm)
#        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#        # l4_points = self.sa4(l3_xyz,l3_points)
#        x = l3_points.view(B, 1024)
#        # x = l4_points.view(B, 1024)
#        x=x.unsqueeze(-1)
#        print('0:',x.shape)



        
#         x = x.unsqueeze(2)
#         #print('05:',x.shape) # [batch_size, 64, 1]
#         B, L, M = x.shape
#         #print('B',B)
#         #print('L',L)
#         #print('M',M)
# #        means = x.mean(1, keepdim=True).detach()
# #        x = x - means
# #        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
# #        x /= stdev
#         x = rearrange(x, 'b l m -> b m l')
#         #print('1:',x.shape)
#         x = self.padding_patch_layer(x)
#         #print('2:',x.shape)
#         x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
#         #print('3:',x.shape)
#         x = rearrange(x, 'b m n p -> (b m) n p')
#         #print('4:',x.shape)
#         outputs = self.in_layer(x)  #[batch_size, 1, 768]
#         #print('5:',outputs.shape)
        batch_size, patches, patch_size = x.size()
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        x = x.view(batch_size, 8, 8, 768)
        x = x.permute(0,3,1,2)
        outputs = self.decoder(x)
            #print('6:',outputs.shape)
       # print('outputs.reshape(B*M, -1):',outputs.reshape(B*M, -1).shape)
        # outputs = outputs.view(outputs.size(0), -1)
        # #print('66:',outputs.shape)
        # outputs = self.out_layer(outputs)
        # #print('7:',outputs.shape)
        # outputs = outputs.view(x.size(0),1,16,16)
        # #print('8:',outputs.shape)
        # outputs = self.conv_layers(outputs)
#         outputs = self.out_layer(outputs.reshape(B * M, -1))
#         # print('7:',outputs.shape)
#         outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
#         # print('8:',outputs.shape)
#         outputs = outputs.view(B, 1, 256, 256)
        #print('9:',outputs.shape)
        #outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        #print('10:',outputs.shape)
        #outputs = outputs.view(B, 3, 256, 256)
        #print('11:',outputs.shape)

#        outputs = outputs * stdev
#        outputs = outputs + means

        return outputs


class GPT_dep(nn.Module): 
    def __init__(self, configs, device,normal_channel=False):
        super(GPT_dep, self).__init__()
    
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
#        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(768, 65536)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # (B, 256, 16, 16) -> (B, 128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            #(B, 128, 32, 32) -> (B, 64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # # (B, 64, 64, 64) -> (B, 32, 128, 128)
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2),
            # # (B, 32, 64, 64) -> (B, 16, 256, 256)
            # nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(0.2),

            # (B, 16, 64, 64) -> (B, 3, 64, 64)
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        
#        self.out_layer = nn.Linear(configs.d_model * self.patch_num, 196608)

#        冻结部分参数
#        if configs.freeze_gptall and configs.pretrain:
#            print('configs.freeze_gptall')
#            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#                param.requires_grad = False
#        elif configs.freeze and configs.pretrain:
#            print('configs.freeze')
#            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#                if 'ln' in name or 'wpe' in name:
#                    param.requires_grad = True
#                else:
#                    param.requires_grad = False

#        各层转移到device上
        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


#    def forward(self, x, itr):
    def forward(self, x):
#        B, _, _ = xyz.shape
#        if self.normal_channel:
#            norm = xyz[:, 3:, :]
#            xyz = xyz[:, :3, :]
#        else:
#            norm = None
#        l1_xyz, l1_points = self.sa1(xyz, norm)
#        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#        # l4_points = self.sa4(l3_xyz,l3_points)
#        x = l3_points.view(B, 1024)
#        # x = l4_points.view(B, 1024)
#        x=x.unsqueeze(-1)
#        print('0:',x.shape)



        
#         x = x.unsqueeze(2)
#         #print('05:',x.shape) # [batch_size, 64, 1]
#         B, L, M = x.shape
#         #print('B',B)
#         #print('L',L)
#         #print('M',M)
# #        means = x.mean(1, keepdim=True).detach()
# #        x = x - means
# #        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
# #        x /= stdev
#         x = rearrange(x, 'b l m -> b m l')
#         #print('1:',x.shape)
#         x = self.padding_patch_layer(x)
#         #print('2:',x.shape)
#         x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
#         #print('3:',x.shape)
#         x = rearrange(x, 'b m n p -> (b m) n p')
#         #print('4:',x.shape)
#         outputs = self.in_layer(x)  #[batch_size, 1, 768]
#         #print('5:',outputs.shape)
        batch_size, patches, patch_size = x.size()
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        x = x.view(batch_size, 8, 8, 768)
        x = x.permute(0,3,1,2)
        outputs = self.decoder(x)
            #print('6:',outputs.shape)
       # print('outputs.reshape(B*M, -1):',outputs.reshape(B*M, -1).shape)
        # outputs = outputs.view(outputs.size(0), -1)
        # #print('66:',outputs.shape)
        # outputs = self.out_layer(outputs)
        # #print('7:',outputs.shape)
        # outputs = outputs.view(x.size(0),1,16,16)
        # #print('8:',outputs.shape)
        # outputs = self.conv_layers(outputs)
#         outputs = self.out_layer(outputs.reshape(B * M, -1))
#         # print('7:',outputs.shape)
#         outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
#         # print('8:',outputs.shape)
#         outputs = outputs.view(B, 1, 256, 256)
        #print('9:',outputs.shape)
        #outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        #print('10:',outputs.shape)
        #outputs = outputs.view(B, 3, 256, 256)
        #print('11:',outputs.shape)

#        outputs = outputs * stdev
#        outputs = outputs + means

        return outputs


class GPT_rgb(nn.Module): 
    def __init__(self, configs, device,normal_channel=False):
        super(GPT_rgb, self).__init__()
    
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
#        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(768, 65536)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # (B, 256, 16, 16) -> (B, 128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            #(B, 128, 32, 32) -> (B, 64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # # (B, 64, 64, 64) -> (B, 32, 128, 128)
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2),
            # # (B, 32, 64, 64) -> (B, 16, 256, 256)
            # nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(0.2),

            # (B, 16, 64, 64) -> (B, 3, 64, 64)
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        
#        self.out_layer = nn.Linear(configs.d_model * self.patch_num, 196608)

#        冻结部分参数
#        if configs.freeze_gptall and configs.pretrain:
#            print('configs.freeze_gptall')
#            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#                param.requires_grad = False
#        elif configs.freeze and configs.pretrain:
#            print('configs.freeze')
#            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#                if 'ln' in name or 'wpe' in name:
#                    param.requires_grad = True
#                else:
#                    param.requires_grad = False

#        各层转移到device上
        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


#    def forward(self, x, itr):
    def forward(self, x):
#        B, _, _ = xyz.shape
#        if self.normal_channel:
#            norm = xyz[:, 3:, :]
#            xyz = xyz[:, :3, :]
#        else:
#            norm = None
#        l1_xyz, l1_points = self.sa1(xyz, norm)
#        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#        # l4_points = self.sa4(l3_xyz,l3_points)
#        x = l3_points.view(B, 1024)
#        # x = l4_points.view(B, 1024)
#        x=x.unsqueeze(-1)
#        print('0:',x.shape)



        
#         x = x.unsqueeze(2)
#         #print('05:',x.shape) # [batch_size, 64, 1]
#         B, L, M = x.shape
#         #print('B',B)
#         #print('L',L)
#         #print('M',M)
# #        means = x.mean(1, keepdim=True).detach()
# #        x = x - means
# #        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
# #        x /= stdev
#         x = rearrange(x, 'b l m -> b m l')
#         #print('1:',x.shape)
#         x = self.padding_patch_layer(x)
#         #print('2:',x.shape)
#         x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
#         #print('3:',x.shape)
#         x = rearrange(x, 'b m n p -> (b m) n p')
#         #print('4:',x.shape)
#         outputs = self.in_layer(x)  #[batch_size, 1, 768]
#         #print('5:',outputs.shape)
        batch_size, patches, patch_size = x.size()
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        x = x.view(batch_size, 8, 8, 768)
        x = x.permute(0,3,1,2)
        outputs = self.decoder(x)
            #print('6:',outputs.shape)
       # print('outputs.reshape(B*M, -1):',outputs.reshape(B*M, -1).shape)
        # outputs = outputs.view(outputs.size(0), -1)
        # #print('66:',outputs.shape)
        # outputs = self.out_layer(outputs)
        # #print('7:',outputs.shape)
        # outputs = outputs.view(x.size(0),1,16,16)
        # #print('8:',outputs.shape)
        # outputs = self.conv_layers(outputs)
#         outputs = self.out_layer(outputs.reshape(B * M, -1))
#         # print('7:',outputs.shape)
#         outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
#         # print('8:',outputs.shape)
#         outputs = outputs.view(B, 1, 256, 256)
        #print('9:',outputs.shape)
        #outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        #print('10:',outputs.shape)
        #outputs = outputs.view(B, 3, 256, 256)
        #print('11:',outputs.shape)

#        outputs = outputs * stdev
#        outputs = outputs + means

        return outputs


if __name__ == '__main__':
#    print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
#print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
#print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
    print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
    print("GPU名称：", torch.cuda.get_device_name(1))    # 根据索引号得到GPU名称
    parser = argparse.ArgumentParser(description='GPT4SCAgrid')
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=1600)
    parser.add_argument('--pred_len', type=int, default=100)
    
    args = parser.parse_args()
    device = torch.device('cuda:0')
    model = GPT4SCAgrid(args, device).to(device)
    print(model)
#    point_cloud = torch.randn(6,3,3000).to(device)
    point_cloud_feature = torch.randn(8,3,10,10).to(device)
    # import ipdb
    # ipdb.set_trace()
    outp = model(point_cloud_feature)
#    print(outp)
    print(outp.shape)

#    point_cloud = torch.randn(1,3,3000)
#    # import ipdb
#    # ipdb.set_trace()
#    feature = model(point_cloud)
#    print(feature)
#    print(feature.shape)


        
'''这段代码是神经网络模型的前向传播（forward）方法的实现。逐行解释代码的功能：

B, L, M = x.shape：获取输入张量 x 的形状信息，其中 B 表示批次大小（batch size），L 表示序列长度（sequence length），M 表示特征维度（feature dimension）。
means = x.mean(1, keepdim=True).detach()：计算输入数据在第一个维度（序列长度维度）上的均值，并使用 detach 方法将其从计算图中分离（detach）。这里使用 keepdim=True 保持维度一致性。
x = x - means：对输入数据减去均值，进行标准化操作。
stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()：计算输入数据在第一个维度上的方差，并使用 torch.sqrt 函数计算标准差。这里添加了一个小的常数 1e-5 以避免除以零的情况发生。同样使用 detach 方法将其从计算图中分离。
x /= stdev：对输入数据进行标准化操作，将其除以标准差。
x = rearrange(x, 'b l m -> b m l')：使用 rearrange 函数将输入数据的维度重排，将维度顺序从 "batch-size, sequence-length, feature-dimension" 调整为 "batch-size, feature-dimension, sequence-length"。
x = self.padding_patch_layer(x)：将输入数据传递给模型中的 padding_patch_layer 层进行处理。
x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)：对输入数据进行展开操作，沿着最后一个维度（sequence-length 维度）进行展开，每个展开窗口的大小为 self.patch_size，步幅为 self.stride。
x = rearrange(x, 'b m n p -> (b m) n p')：使用 rearrange 函数将展开后的数据进行维度重排，将维度顺序从 "batch-size, feature-dimension, window-size, patch-size" 调整为 "(batch-size * feature-dimension), window-size, patch-size"。
outputs = self.in_layer(x)：将展开后的数据传递给模型中的 in_layer 层进行处理，得到输出。
if self.is_gpt: outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state：如果模型类型是 GPT（self.is_gpt 为 True），则将输出数据传递给 GPT 模型（self.gpt2）进行处理，获取最后一个隐藏状态（last hidden state）作为输出。
outputs = self.out_layer(outputs.reshape(B*M, -1))：将输出数据进行形状重排，将维度从 "(batch-size * feature-dimension), window-size, patch-size" 调整为 "(batch-size * feature-dimension), output-size"，然后将其传递给模型中的 out_layer 层进行处理。
outputs = rearrange(outputs, '(b m) l -> b l m', b=B)：使用 rearrange 函数将输出数据的维度重排，将维度顺序从 "(batch-size * feature-dimension), output-size" 调整为 "batch-size, output-size, feature-dimension"，并恢复批次大小为原始值。
outputs = outputs * stdev：将输出数据乘以标准差，进行逆标准化操作。
outputs = outputs + means：将输出数据加上均值，进行逆标准化操作。
return outputs：返回最终的输出结果。'''
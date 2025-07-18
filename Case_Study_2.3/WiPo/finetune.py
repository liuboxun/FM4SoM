from models.head import csi_recon, img_recon
from models.tokenizer import csi_tokenizer, img_tokenizer
from models.utils import Adapter
from models.wipo import WiPo_finetune

import random
import math

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


img_size = 32
batch_size = 64
channels_finetune = 16
patch_size_img = 8
width = 384
layers = 6
snr_min = 0
snr_max = 18

# Finetune on image reconstruction
img_tokenizer = img_tokenizer(width=width, patch_size_img=patch_size_img).cuda()
img_recon = img_recon(width=width, img_size=img_size, patch_size_img=patch_size_img).cuda()
wipo_finetune = WiPo_finetune(width=width, layers=layers, channels_finetune=channels_finetune, adapter_dim=32).cuda()

# Finetune on CIFAR10
preprocess = transforms.Compose([
    # transforms.RandomCrop(img_size),  # 随机裁剪
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# 加载 CIFAR-100 数据集
train_img = CIFAR10(root='../MetaTransformer/data', train=True, download=True, transform=preprocess)
test_img = CIFAR10(root='../MetaTransformer/data', train=False, download=True, transform=preprocess)
# 构造 DataLoader
train_loader_img = DataLoader(train_img, batch_size=batch_size, shuffle=True)
# train_loader_img = DataLoader(train_img, batch_size=batch_size, shuffle=True)
test_loader_img = DataLoader(test_img, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
criterion_test = nn.MSELoss(reduction='none')

def compute_img_psnr(img, recon_img):
    loss = criterion_test(recon_img, img)
    loss = torch.mean(loss, dim=[1,2])
    loss = torch.mean(10 * torch.log10(1 / (loss + 1e-7)))
    return loss.item() 

max_lr = 4e-3
min_lr = 1e-6


# 冻结所有层的参数
for param in wipo_finetune.parameters():
    param.requires_grad = False

for param in wipo_finetune.encoder_img.parameters():
    param.requires_grad = True
for param in wipo_finetune.decoder_img.parameters():
    param.requires_grad = True

# 将LayerNorm层的参数设为可训练
for name, layer in wipo_finetune.named_modules():
    if isinstance(layer, nn.RMSNorm):
        for param in layer.parameters():
            param.requires_grad = True

# 将LayerNorm层的参数设为可训练
for name, layer in wipo_finetune.backbone.named_modules():
    if isinstance(layer, nn.RMSNorm):
        for param in layer.parameters():
            param.requires_grad = True
    if isinstance(layer, Adapter):
        for param in layer.parameters():
            param.requires_grad = True
    if 'snr' in name:
        for param in layer.parameters():
            param.requires_grad = True
            

params_to_optimize = [
    {'params': img_recon.parameters(), 'lr': max_lr},  
    {'params': img_tokenizer.parameters(), 'lr': max_lr},
    {'params': wipo_finetune.parameters(), 'lr': max_lr / 5}
]

warmup_epochs = 5
num_epochs = 100
print_freq = 50

class WarmupCosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_steps = warmup_epochs
        self.total_steps = total_epochs
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linearly scale the learning rate
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            # print(lr_scale)
        else:
            # Cosine decay phase
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
            lr_scale = cosine_decay
            # print(lr_scale)

        return [base_lr * lr_scale for base_lr in self.base_lrs]
    
# 使用示例
optimizer = torch.optim.AdamW(params_to_optimize)
scheduler = WarmupCosineDecayScheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=num_epochs)

trainables = [p for p in wipo_finetune.parameters() if p.requires_grad]
print('WiPo trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

def train(snr_input, mode):
    running_loss = 0.0
    running_loss_csi = 0.0
    running_loss_img = 0.0
    running_loss_point = 0.0
    best_psnr = -1
    
    for epoch in range(0, num_epochs):
        iters = 0
        # scheduler.step()
        for data_img in train_loader_img:
            img, _ = data_img
        
            # Image Reconstruction
            img = img.cuda()
            img_token = img_tokenizer(img)
            if(mode == 'fixed'):
                snr = snr_input
            elif(mode == 'dynamic'):
                snr = random.uniform(snr_min, snr_max)
            feats_img, true_tokens, recon_tokens = wipo_finetune(img_token, snr, 2)
            recon_img = img_recon(feats_img)
            loss_img = criterion(recon_img, img)
            loss_img = loss_img
            
            optimizer.zero_grad()   
            loss_img.backward()
            optimizer.step()
                    
            running_loss_img += loss_img.item()
            
            iters += 1
            if(iters % print_freq == 0):
                print("Epoch: ", epoch, ", Iters: ", iters, ", image task loss: ", running_loss_img / print_freq, ", csi task loss: ", running_loss_csi / print_freq, ", point task loss: ", running_loss_point / print_freq)
                running_loss = 0
                running_loss_csi = 0
                running_loss_img = 0
                running_loss_point = 0
        
        scheduler.step()
        running_loss_csi = 0
        running_loss_img = 0
        running_loss_point = 0
        
        csi_nmse = 0
        img_psnr = 0
        point_distance = 0
        counts = 0  
        for data_img in test_loader_img:
            with torch.no_grad():
                img, _ = data_img
                
                # Image Reconstruction
                img = img.cuda()
                img_token = img_tokenizer(img)
                if(mode == 'fixed'):
                    snr = snr_input
                elif(mode == 'dynamic'):
                    snr = random.uniform(snr_min, snr_max)
                feats_img, _, _ = wipo_finetune(img_token, snr, 2)
                recon_img = img_recon(feats_img)
                
                img_psnr += (compute_img_psnr(img, recon_img))
                counts += 1
        print("Epoch: ", epoch, ",CSI NMSE: ", csi_nmse / counts, "Image PSNR: ", img_psnr / counts, "Point Distance: ", point_distance / counts)

        torch.save(
                    {
                        'img_tokenizer': img_tokenizer.state_dict(),
                        'wipo_finetune': wipo_finetune.state_dict(),
                        'img_recon': img_recon.state_dict(),
                    }, 'cifar_finetuned_base_c'+str(channels_finetune)+'_d'+str(width)+'_'+mode+'.pth'
                )

checkpoint = torch.load('checkpoints/wipo_img+csi+point_swin_L'+str(layers)+'_D'+str(width)+'.pth', map_location=torch.device('cuda'))
wipo_finetune.load_state_dict(checkpoint['backbone'], strict=False)
train(snr_input = None, mode = 'dynamic')
import torch.nn as nn
import torch
from models.utils import unpatchify_img

class csi_recon(nn.Module):
    def __init__(self, width, patch_size_csi, antennas):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(width, antennas * patch_size_csi))
        
    def forward(self, x):
        return self.head(x)
    
class img_recon(nn.Module):
    def __init__(self, width=512, img_size=32, patch_size_img=8):
        super(img_recon, self).__init__()
        self.patch_size_img = patch_size_img
        self.patch_num = (img_size // patch_size_img) ** 2
        self.decoder_pred = nn.Sequential(nn.Linear(width, patch_size_img**2 * 3, bias=True))
        
    def forward(self, x):
        B, _, _, = x.shape
        x = self.decoder_pred(x)   
        # clamp the image value to [0, 1]
        x = torch.clamp(x, min=0, max=1)
        x = unpatchify_img(x, self.patch_size_img)
        return x
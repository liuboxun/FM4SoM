import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding1D, self).__init__()
        self.d_model = d_model

        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, C]
        """
        x = x + self.pe[:, :x.size(1), :].cuda()
        return x

class csi_tokenizer(nn.Module):
    def __init__(self, width=512, patch_size_csi=8, subcarriers=32):
        super(csi_tokenizer, self).__init__()
        scale = width ** -0.5
        self.pos_encoder = SinusoidalPositionalEncoding1D(d_model=width)
        self.csi_conv = nn.Conv1d(in_channels=32, out_channels=width, kernel_size=patch_size_csi, stride=patch_size_csi)
    def forward(self, x):
        x = self.csi_conv(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        return x

class img_tokenizer(nn.Module):
    def __init__(self, width=512, patch_size_img=8):
        super(img_tokenizer, self).__init__()
        scale = width ** -0.5
        self.img_conv = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size_img, stride=patch_size_img, bias=False)
        self.pos_encoder = SinusoidalPositionalEncoding1D(width)
    def forward(self, x):
        x = self.img_conv(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        return x
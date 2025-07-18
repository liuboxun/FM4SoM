import torch.nn as nn
import torch
from models.utils import SwinTransformerBlockSnrAdapter
from models.channel import Channel

class WiPo_finetune(nn.Module):
    def __init__(self, width, layers, channels_finetune, adapter_dim):
        super().__init__()
        self.backbone = nn.Sequential(*[SwinTransformerBlockSnrAdapter(dim=width, adapter_dim = adapter_dim, num_heads=4, window_size=4, norm_layer=nn.RMSNorm, shift_size=0 if (i % 2 == 0) else 2) for i in range(layers)])
        self.layers = layers
        self.ln_pre = nn.RMSNorm(width)
        self.ln_post = nn.RMSNorm(width)
        self.channel = Channel(chan_type=1)        
        self.encoder_img = nn.Linear(width, channels_finetune)
        self.decoder_img = nn.Linear(channels_finetune, width)

    def forward(self, x, snr, modality):
        B, N, _ = x.shape
        x = self.ln_pre(x)
        for i in range(0, self.layers // 2):
            x = self.backbone[i](x, snr)
        
        true_tokens = x.clone()
        x = self.encoder_img(x)
        x = self.channel.forward(x, snr, False)
        x = self.decoder_img(x)
        recon_tokens = x.clone()
        for i in range(self.layers // 2, self.layers):
            x = self.backbone[i](x, snr)

        x = self.ln_post(x)
        return x, true_tokens, recon_tokens
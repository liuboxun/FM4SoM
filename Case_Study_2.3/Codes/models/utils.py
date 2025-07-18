import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def unpatchify_img(patches, patch_size):
    p = patch_size
    n, num_patches, patch_dim = patches.shape
    # 根据 patch 数量计算每个维度的 patch 个数
    h = w = int(math.sqrt(num_patches))
    
    # 将 patches 重塑回 (n, h, w, p, p, 3) 的形状
    x = patches.reshape(n, h, w, p, p, 3)
    
    # 逆转 patchify 中的 einsum 操作：从 (n, h, w, p, p, 3) 转换回 (n, 3, h, p, w, p)
    x = torch.einsum('nhwpqc->nchpwq', x)
    
    # 重新排列得到原始图像，形状为 (n, 3, h*p, w*p)
    imgs = x.reshape(n, 3, h * p, w * p)
    return imgs


def unpatchify_csi(x, patch_size, antennas, subcarriers):
    # print(x.shape)
    p = patch_size
    m = antennas
    s = subcarriers * 2
    l = x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], l, p, m))
    x = torch.einsum('nlpm->nmlp', x)
    imgs = x.reshape(shape=(x.shape[0], m, s))
    return imgs

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    # copied from timm/models/layers/drop.py
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    # copied from timm/models/layers/drop.py
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        L (int): Length of data

    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.contiguous().view(B, L, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The length of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (int): The length of the window in pre-training.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias=True, attn_drop=0., proj_drop=0., pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(1, 512, bias=True),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = relative_coords_w  # 2*W-1
        # scale = pretrained_window_size - 1 if pretrained_window_size > 0 else window_size - 1
        # relative_coords_table = relative_coords_table / scale  # normalize to -1, 1
        if pretrained_window_size > 0:
            relative_coords_table[:] /= (pretrained_window_size - 1)
        else:
            relative_coords_table[:] /= (self.window_size - 1)
        relative_coords_table = relative_coords_table * 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table.unsqueeze(1), persistent=False)  # (2*W-1, 1)

        # get pair-wise relative position index for each token inside the window
        coords_w = torch.arange(self.window_size)
        relative_coords = coords_w[:, None] - coords_w[None, :]  # W, W
        relative_coords = relative_coords + self.window_size - 1  # shift to start from 0
        # relative_position_index | example
        # [2, 1, 0]
        # [3, 2, 1]
        # [4, 3, 2]
        self.register_buffer("relative_position_index", relative_coords, persistent=False)  # (W, W): range of 0 -- 2*(W-1)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        device = next(self.parameters()).device
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        # 这里看起来使用了QK-Norm
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)  # (2*W-1, nH)
        
        # idx = self.relative_position_index.flatten().clone()          # (W*W, )
        # relative_position_bias = relative_position_bias_table[idx]    # (W*W, nH)
        # relative_position_bias = (
        #     relative_position_bias
        #         .view(self.window_size, self.window_size, -1)          # (W, W, nH)
        #         .permute(2, 0, 1)                                      # (nH, W, W)
        #         .contiguous()
        # )
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # (W, W, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, W, W
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # print("Attention Score: ", attn)
        # print("Relative Position: ", relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=pretrained_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, L, C = x.shape
        assert L >= self.window_size, f'input length ({L}) must be >= window size ({self.window_size})'
        assert L % self.window_size == 0, f'input length ({L}) must be divisible by window size ({self.window_size})'

        shortcut = x

        # zero-padding shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            shifted_x[:, -self.shift_size:] = 0.
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # (B, L, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
            x[:, :self.shift_size] = 0.  # remove invalid embs
        else:
            x = shifted_x

        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, L: int):
        flops = 0
        # norm1
        flops += self.dim * L
        # W-MSA/SW-MSA
        nW = L / self.window_size
        flops += nW * self.attn.flops(self.window_size)
        # mlp
        flops += 2 * L * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * L
        return flops

class SwinTransformerBlockSnr(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=pretrained_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.pool2d = nn.AdaptiveAvgPool2d((1, dim))
        self.snr_adap = nn.Sequential(
            nn.Linear(dim + 1, dim // 16),
            nn.ReLU(),
            nn.Linear(dim // 16, dim),
            nn.Sigmoid()
        )
        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, snr, pass_channel=False):
        B, L, C = x.shape
        assert L >= self.window_size, f'input length ({L}) must be >= window size ({self.window_size})'
        assert L % self.window_size == 0, f'input length ({L}) must be divisible by window size ({self.window_size})'

        shortcut = x

        # zero-padding shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            shifted_x[:, -self.shift_size:] = 0.
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # (B, L, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
            x[:, :self.shift_size] = 0.  # remove invalid embs
        else:
            x = shifted_x

        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        if(pass_channel == True):
            snr = torch.tensor(snr).repeat(x.shape[0], 1, 1).cuda()
            x_pool = self.pool2d(x)
            x_cat = torch.cat([x_pool, snr], dim = 2)
            attn = self.snr_adap(x_cat)
            x = x + attn * x
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, L: int):
        flops = 0
        # norm1
        flops += self.dim * L
        # W-MSA/SW-MSA
        nW = L / self.window_size
        flops += nW * self.attn.flops(self.window_size)
        # mlp
        flops += 2 * L * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * L
        return flops

class Adapter(nn.Module):
    def __init__(self, width, dim):
        super().__init__()
        self.down_project = nn.Linear(width, dim)
        self.gelu = nn.GELU()
        self.up_project = nn.Linear(dim, width)
        self.initialize_weights(self.down_project)
        self.initialize_weights(self.up_project)
    def forward(self, x):
        y = self.down_project(x)
        y = self.gelu(y)
        y = self.up_project(y) + x
        return y
    
    def initialize_weights(self, m):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(m.bias, 0)

class SwinTransformerBlockSnrAdapter(nn.Module):
    r""" Swin Transformer Block with Adapter.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, adapter_dim = 32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=pretrained_window_size)
        self.adapter_1 = Adapter(dim, adapter_dim)
        self.adapter_2 = Adapter(dim, adapter_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.pool2d = nn.AdaptiveAvgPool2d((1, dim))
        self.snr_adap = nn.Sequential(
            nn.Linear(dim + 1, dim // 16),
            nn.ReLU(),
            nn.Linear(dim // 16, dim),
            nn.Sigmoid()
        )
        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, snr):
        B, L, C = x.shape
        assert L >= self.window_size, f'input length ({L}) must be >= window size ({self.window_size})'
        assert L % self.window_size == 0, f'input length ({L}) must be divisible by window size ({self.window_size})'

        shortcut = x

        # zero-padding shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            shifted_x[:, -self.shift_size:] = 0.
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # (B, L, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
            x[:, :self.shift_size] = 0.  # remove invalid embs
        else:
            x = shifted_x

        snr = torch.tensor(snr).repeat(x.shape[0], 1, 1).cuda()
        x = shortcut + self.drop_path(self.adapter_1(self.norm1(x)))

        # FFN
        x = x + self.drop_path(self.adapter_2(self.norm2(self.mlp(x))))

        # snr = torch.tensor(snr).repeat(x.shape[0], 1, 1).cuda()
        x_pool = self.pool2d(x)
        x_cat = torch.cat([x_pool, snr], dim = 2)
        attn = self.snr_adap(x_cat)
        x = attn * x + x
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, L: int):
        flops = 0
        # norm1
        flops += self.dim * L
        # W-MSA/SW-MSA
        nW = L / self.window_size
        flops += nW * self.attn.flops(self.window_size)
        # mlp
        flops += 2 * L * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * L
        return flops

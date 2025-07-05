# coding=utf-8
from functools import partial

import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp
from einops import rearrange

from Embed import DataEmbedding, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid_with_resolution

from mask_strategy import *
import copy
import time
from scipy import interpolate

from transformers import ViTModel
import torchvision.models as models
def WiFo_model(args, **kwargs):

    if args.size == 'small':
        model = WiFo(
            embed_dim=256,
            depth=6,
            decoder_embed_dim = 256,
            decoder_depth=4, # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model
    elif args.size == 'little':
        model = WiFo(
            embed_dim=128,
            depth=6,
            decoder_embed_dim = 128,
            decoder_depth=4, # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model
    elif args.size == 'tiny':
        model = WiFo(
            embed_dim=64,
            depth=6,
            decoder_embed_dim = 64,
            decoder_depth=4, # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model
    elif args.size == 'base':
        model = WiFo(
            embed_dim=512,
            depth=6,
            decoder_embed_dim = 512,
            decoder_depth=4, # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.mlp2(self.relu(self.mlp1(x)))

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim, bias= qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class WiFo(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_size=1, in_chans=1,
                 embed_dim=512, decoder_embed_dim=512, depth=12, decoder_depth=8, num_heads=8,  decoder_num_heads=4,
                 mlp_ratio=2, norm_layer=nn.LayerNorm, t_patch_size=1,
                 no_qkv_bias=False, pos_emb = 'trivial', args=None, ):
        super().__init__()

        self.args = args

        self.pos_emb = pos_emb

        self.Embedding = DataEmbedding(2, embed_dim, args=args)
        # mask
        self.t_patch_size = t_patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        self.embed_dim = embed_dim
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 64, embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 4, embed_dim)
        )
        self.decoder_pos_embed_fre = nn.Parameter(torch.zeros(1,32,embed_dim))
        self.decoder_pos_embed_antenna = nn.Parameter(torch.zeros(1,4,embed_dim))
        self.decoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 64, decoder_embed_dim)
        )
        self.decoder_pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 4,  decoder_embed_dim)
        )

        
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.no_qkv_bias = no_qkv_bias
        self.norm_layer = norm_layer


        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias= not self.args.no_qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_patch_size * patch_size**2 * 2,
            bias=True,
        )

        self.initialize_weights_trivial()

        # Pre-trained res18 as feature extractor
        self.res18 = models.resnet18(pretrained=True)
        self.res18.eval()
        self.res18_align_layer = nn.Linear(1000, decoder_embed_dim)

        print("model initialized!")

    def init_emb(self):
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w = self.Embedding.value_embedding.tokenConv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)


    def get_weights_sincos(self, num_t_patch, num_patch_1, num_patch_2):

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1 = num_patch_1,
            grid_size2 = num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
                torch.zeros(1, num_patch_1 * num_patch_2, self.embed_dim)
            )
        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.embed_dim)
        )

        pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(pos_embed_temporal.shape[-1], np.arange(num_t_patch, dtype=np.float32))

        pos_embed_temporal.data.copy_(torch.from_numpy(pos_temporal_emb).float().unsqueeze(0))

        pos_embed_spatial.requires_grad = False
        pos_embed_temporal.requires_grad = False

        return pos_embed_spatial, pos_embed_temporal, copy.deepcopy(pos_embed_spatial), copy.deepcopy(pos_embed_temporal)

    def initialize_weights_trivial(self):
        std_pre = 0.02
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=std_pre)


        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=std_pre)


        w = self.Embedding.value_embedding.tokenConv.weight.data

        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=std_pre)
        #torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    



    def patchify(self, imgs):
        """
        imgs: (N, 1, T, H, W)
        x: (N, L, patch_size**2 *1)
        # The input is imgs containing real and imaginary parts, and the output is complex imgs
        """
        N, _, T, H, W = imgs.shape  # N, 2, T, H, W
        p = self.args.patch_size
        u = self.args.t_patch_size
        assert H % p == 0 and W % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u
        x = imgs.reshape(shape=(N, 2, t, u, h, p, w, p))  # 第2维代表实部/虚部
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 2))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        x = x[:,:,:u * p**2] + 1j*x[:,:,u * p**2:]
        # x = x[:, :, ::2] + 1j * x[:, :, 1::2]
        return x


    def unpatchify(self, imgs):
        """
        imgs: (N, L, patch_size**2 *1)
        x: (N, 1, T, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info
        imgs = torch.cat([imgs.real, imgs.imag], dim=-1)   # N, L, patch_size**3*2
        imgs = rearrange(imgs, 'b (t h w) (u p q o)-> b (t u) (h p) (w q) o', t=t,h=h,w=w,u=u,p=p,q=p,o=2)
        imgs = torch.complex(imgs[..., 0], imgs[..., 1])
        # imgs = imgs.reshape(shape=(N, t, h, w, u, p, p))
        # imgs = torch.einsum("nthwupq->ntuhpwq", imgs)
        # imgs = imgs.reshape(shape=(N, T, H, W))
        return imgs


    def pos_embed_enc(self, ids_keep, batch, input_size):

        if self.pos_emb == 'trivial':  # [1,256,384]
            pos_spatial = self.pos_embed_fre[:,:input_size[2]].repeat(
                1, input_size[1],1
            ) + torch.repeat_interleave(
                self.pos_embed_antenna[:,:input_size[1]],
                input_size[2],
                dim=1
            )  # [1,64,384]
            pos_embed = pos_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.pos_emb == 'SinCos':
            pos_embed_spatial, pos_embed_temporal, _, _ = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])

            pos_embed = pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )  # [1,256,256]

        pos_embed = pos_embed.to(ids_keep.device)

        pos_embed = pos_embed.expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )
        return pos_embed_sort

    def pos_embed_enc_3d(self, ids_keep, batch, input_size, scale):

        T, H, W = input_size
        t = torch.arange(T)
        h = torch.arange(H)
        w = torch.arange(W)

        tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')

        ED = self.embed_dim
        if ED == 256:
            ED1 = 86
            ED2 = 86
            ED3 = 84
        elif ED == 128:
            ED1 = 42
            ED2 = 42
            ED3 = 44
        elif ED ==768:
            ED1 = 256
            ED2 = 256
            ED3 = 256
        elif ED == 512:
            ED1 = 170
            ED2 = 170
            ED3 = 172
        elif ED == 64:
            ED1 = 22
            ED2 = 22
            ED3 = 20

        emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(ED1, tt.flatten(),scale[0])
        emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(ED2, hh.flatten(),scale[1])
        emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(ED3, ww.flatten(),scale[2])

        pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)
        pos_embed = torch.from_numpy(pos_embed).float()
        pos_embed.requires_grad = False

        pos_embed = pos_embed.to(ids_keep.device)  # [1,256,256]
        pos_embed = pos_embed.expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )
        return pos_embed_sort



    def pos_embed_dec(self, ids_keep, batch, input_size):

        if self.pos_emb == 'trivial':
            pos_spatial = self.decoder_pos_embed_fre[:, :input_size[2]].repeat(
                1, input_size[1], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_antenna[:, :input_size[1]],
                input_size[2],
                dim=1
            )
            decoder_pos_embed = pos_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.pos_emb == 'SinCos':
            _, _, decoder_pos_embed_spatial, decoder_pos_embed_temporal  = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])

            decoder_pos_embed = decoder_pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                decoder_pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        decoder_pos_embed = decoder_pos_embed.to(ids_keep.device)

        decoder_pos_embed = decoder_pos_embed.expand(batch, -1, -1)

        return decoder_pos_embed

    def pos_embed_dec_3d(self, ids_keep, batch, input_size, scale):

        T, H, W = input_size
        t = torch.arange(T)
        h = torch.arange(H)
        w = torch.arange(W)

        tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')

        ED = self.embed_dim
        if ED == 256:
            ED1 = 86
            ED2 = 86
            ED3 = 84
        elif ED == 128:
            ED1 = 42
            ED2 = 42
            ED3 = 44
        elif ED ==768:
            ED1 = 256
            ED2 = 256
            ED3 = 256
        elif ED == 512:
            ED1 = 170
            ED2 = 170
            ED3 = 172
        elif ED == 64:
            ED1 = 22
            ED2 = 22
            ED3 = 20

        emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(ED1, tt.flatten(),scale[0])
        emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(ED2, hh.flatten(),scale[1])
        emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(ED3, ww.flatten(),scale[2])

        pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)
        decoder_pos_embed = torch.from_numpy(pos_embed).float()
        decoder_pos_embed.requires_grad = False

        decoder_pos_embed = decoder_pos_embed.to(ids_keep.device)

        decoder_pos_embed = decoder_pos_embed.expand(batch, -1, -1)

        return decoder_pos_embed


    def forward_encoder(self, x, mask_ratio, mask_strategy, seed=None, data=None, mode='backward', scale=None):
        # embed patches
        N, _, T, H, W = x.shape
        x = self.Embedding(x)
        _, L, C = x.shape

        T = T // self.args.t_patch_size
        H = H // self.patch_size
        W = W // self.patch_size


        if mask_strategy == 'random':
            x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)

        elif mask_strategy == 'temporal':
            x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T)

        elif mask_strategy == 'fre':
            x, mask, ids_restore, ids_keep = fre_masking(x, mask_ratio, T=T, H=H, W=W)


        input_size = (T, H, W)
        if self.pos_emb == 'SinCos' or self.pos_emb == 'trivial':
            pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)  # PE
            assert x.shape == pos_embed_sort.shape
            x_attn = x + pos_embed_sort
        elif self.pos_emb == 'SinCos_3D':
            pos_embed_sort = self.pos_embed_enc_3d(ids_keep, N, input_size, scale=scale)
            assert x.shape == pos_embed_sort.shape
            x_attn = x + pos_embed_sort
        elif self.pos_emb == 'None':
            x_attn = x
        # apply Transformer blocks
        for index, blk in enumerate(self.blocks):
            x_attn = blk(x_attn)

        x_attn = self.norm(x_attn)

        return x_attn, mask, ids_restore, input_size

    def forward_decoder(self, x, ids_restore, mask_strategy, input_size=None, data=None, scale=None, rgb=None):
        N = x.shape[0]
        T, H, W = input_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        if mask_strategy == 'random':
            x = random_restore(x, ids_restore, N, T,  H, W, C, self.mask_token)


        elif mask_strategy == 'temporal':
            x = causal_restore(x, ids_restore, N, T, H,  W, C, self.mask_token)

        elif mask_strategy == 'fre':
            x = fre_restore(x, ids_restore, N, T, H,  W, C, self.mask_token)



        if self.pos_emb == 'SinCos' or self.pos_emb == 'trivial':
            decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size)  # PE
            assert x.shape == decoder_pos_embed.shape
            x_attn = x + decoder_pos_embed
        elif self.pos_emb == 'SinCos_3D':
            decoder_pos_embed = self.pos_embed_dec_3d(ids_restore, N, input_size, scale=scale)
            assert x.shape == decoder_pos_embed.shape
            x_attn = x + decoder_pos_embed
        elif self.pos_emb == 'None':
            x_attn = x

        if rgb is not None:
            x_attn = torch.cat([rgb, x_attn], dim=1)

        # apply Transformer blocks
        for index, blk in enumerate(self.decoder_blocks):
            x_attn = blk(x_attn)

        if rgb is not None:
            x_attn = x_attn[:, 1:, ...]   # remove [img]

        x_attn = self.decoder_norm(x_attn)

        return x_attn


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 2, T, H, W] 1000*1*12*16*96
        pred: [N, t*h*w, u*p*p*1] 1000*576*32(2*4*4)
        mask: [N*t, h*w], 0 is keep, 1 is remove, 1000*576
        """

        target = self.patchify(imgs)


        assert pred.shape == target.shape

        loss = torch.abs(pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)
        loss1 = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss2 = (loss * (1-mask)).sum() / (1-mask).sum()
        return loss1, loss2, target


    def forward(self, imgs, rgb=None, mask_ratio=0.5, mask_strategy='random', seed=None, data='none',
                time_output=False):
        # imgs: [B, 2, N, K]
        # imgs = torch.stack(imgs).squeeze(1)
        if len(imgs.shape) == 4:
            imgs = imgs.unsqueeze(2).repeat(1, 1, 4, 1, 1)

        snr_db = 20
        noise_power = torch.mean(imgs ** 2) * 10 ** (-snr_db / 10)
        noise = torch.randn_like(imgs) * torch.sqrt(noise_power)
        imgs_n = imgs + noise

        scale = [1,1,1]
        start_time = time.time()
        T, H, W = imgs_n.shape[2:]
        latent, mask, ids_restore, input_size = self.forward_encoder(imgs_n, mask_ratio, mask_strategy, seed=seed, data=data, mode ='backward', scale=scale)
        if rgb is not None:
            rgb_feature = self.res18_align_layer(self.res18(rgb)).unsqueeze(1)
            pred = self.forward_decoder(latent, ids_restore, mask_strategy, input_size=input_size, data=data, scale=scale, rgb=rgb_feature)
        else:
            pred = self.forward_decoder(latent, ids_restore, mask_strategy, input_size = input_size, data=data,scale=scale)  # [N, L, p*p*1]

        # predictor projection
        pred = self.decoder_pred(pred)

        time_all = time.time() - start_time


        Len = self.t_patch_size * self.patch_size ** 2
        pred_complex = pred[:, :, :Len] + 1j * pred[:, :, Len:]


        loss1, loss2, target = self.forward_loss(imgs_n, pred_complex, mask)

        if time_output:
            return loss1, loss2, pred_complex, target, mask, time_all
        else:
            return loss1, loss2, pred_complex, target, mask





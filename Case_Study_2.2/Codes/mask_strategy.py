# coding=utf-8
import torch
import numpy as np

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    x_masked是x[ids_keep]
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1
    )  # 128 576 ascend: small is keep, large is remove 获得一个排序 相当于x[ids_shuffle]可以获得排序后的x
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # 128*576 反排序 相当于x[ids_restore]=x

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # 128*288
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # 128*288*128

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)  # 128*576
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # ids_restore的第i个元素决定mask第i个元素是0/1

    return x_masked, mask, ids_restore, ids_keep

def causal_masking(x, mask_ratio, T):
    N, L, D = x.shape # batch, length, dim
    x = x.reshape(N, T, L//T, D)
    N, T, L, C = x.shape

    len_keep = int(T * (1 - mask_ratio))


    noise = torch.arange(T).unsqueeze(dim=0).repeat(N,1)
    noise = noise.to(x)  # 转到设备上

    ids_shuffle = torch.argsort(
        noise, dim=1
    )  # ascend: small is keep, large is remove N*T
    ids_restore = torch.argsort(ids_shuffle, dim=1) # N*T

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # N*T/2
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(2).unsqueeze(-1).repeat(1, 1, L, D))

    assert (x_masked == x[:,:len_keep]).all()

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, T, L], device=x.device)  # 大小是N*T*L
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore.unsqueeze(2).repeat(1,1,L)).reshape(N,-1)

    ids_keep = ids_keep.unsqueeze(2).repeat(1,1,L).reshape(N,-1)
    x_masked = x_masked.reshape(N, -1, x_masked.shape[-1])

    return x_masked, mask, ids_restore, ids_keep

def fre_masking(x, mask_ratio, T, H, W):
    N, L, D = x.shape # batch, length, dim
    x = x.reshape(N, T, H, W, D)

    len_keep = int(W * (1 - mask_ratio))


    noise = torch.arange(W).unsqueeze(dim=0).unsqueeze(0).unsqueeze(0).repeat(N,T,H,1) # N*T*H*W
    noise = noise.to(x)  # 转到设备上

    ids_shuffle = torch.argsort(
        noise, dim=3
    )  # ascend: small is keep, large is remove N*T
    ids_restore = torch.argsort(ids_shuffle, dim=3)  # N*T*H*W

    # keep the first subset
    ids_keep = ids_shuffle[:, :,:, :len_keep]  # N*T*H*W/2
    x_masked = torch.gather(x, dim=3, index=ids_keep.unsqueeze(4).repeat(1, 1, 1, 1, D))  #

    assert (x_masked == x[:,:,:,:len_keep]).all()

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, T, H, W], device=x.device)
    mask[:,:,:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=3, index=ids_restore).reshape(N,-1)  # N*T*H*W

    ids_keep = ids_keep.reshape(N,-1)
    x_masked = x_masked.reshape(N, -1, x_masked.shape[-1])

    return x_masked, mask, ids_restore, ids_keep




def random_restore(x, ids_restore, N, T, H, W, C, mask_token):
    # ids_restore: [N, T*H*W]
    mask_tokens = mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)  # mask_token是一个learnable的参数，[1,1,C]
    x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
    x_ = x_.view([N, T * H * W, C])
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
    )  # unshuffle
    x = x_.view([N, T * H * W, C])

    return x

def causal_restore(x, ids_restore, N, T, H, W, C, mask_token):
    # ids_restore: [N,T]
    x = x.reshape(N, -1, H * W, x.shape[-1])
    mask_tokens = mask_token.repeat(N, T - x.shape[1] , H * W, 1)
    x_ = torch.cat([x, mask_tokens], dim=1) 
    x_ = torch.gather(x_, dim=1, index = ids_restore.unsqueeze(2).unsqueeze(-1).repeat(1, 1, H * W, x_.shape[-1]))
    x = x_.view([N, T * H * W, C])
    return x

def fre_restore(x, ids_restore, N, T, H, W, C, mask_token):
    # ids_restore: [N,T,H,W]
    x = x.reshape(N, T, H, -1, x.shape[-1])
    mask_tokens = mask_token.repeat(N, T, H, W-x.shape[3],1)
    x_ = torch.cat([x, mask_tokens], dim=3)
    x_ = torch.gather(x_, dim=3, index=ids_restore.unsqueeze(-1).repeat(1,1,1,1,x_.shape[-1]))
    x = x_.view([N, T*H*W, C])
    return x

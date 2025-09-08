import glob
import numpy as np
import torch
from einops import rearrange
import torch_dct as dct

DATA_PATH = "/data1/share_data/purdue/patch/32x32/"
# DATA_PATH = "/data1/share_data/purdue/fullpatch/32x32/"

# 扫描所有pkl文件
def file_scanf():
    return np.array(glob.glob(DATA_PATH + "/*.pkl"))

def group_replace_patch(x, ratio=0.2):
    group,b, t, h, w = x.shape
    N = int(ratio * t)
    # 生成随机索引
    indices = torch.randperm(t)[:N]
    patches = dct.dct_2d(x[:,:, indices])
    patches = time_norm(patches,dim=2)
    x[:,:, indices] = patches
    return x



def replace_patch(x, ratio=0.2):
    b, t, h, w = x.shape
    N = int(ratio * t)
    # 生成随机索引
    indices = torch.randperm(t)[:N]
    patches = dct.dct_2d(x[:, indices])
    patches = time_norm(patches)
    x[:, indices] = patches
    return x


def normalize_samples(x):
    # 计算每个样本的均值和标准差
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)

    # 进行归一化
    x_normalized = (x - mean) / (std + 1e-6)

    return x_normalized

#   均匀采样
# def x_group(x, group=8):
#     # x.shape = (b, 2048, 32, 32)
#     # b = x.shape[0]
#     b, t, h, w = x.shape
#     x= rearrange(x, 'b (ng group)  h w  ->group b ng h w',group=group)
#     # Step 3: reshape to (d, b, 64, 8, 32, 32)
#     x = x.view(group, b, 64, t//(group*64), 32, 32)
#     # Step 4: mean across dimension 3 -> (d, b, 64, 1, 32, 32)
#     x = x.mean(dim=3, keepdim=True)
    
#     # Step 5: reshape to (d, b, 64, 32, 32)
#     x = x.view(group, b, 64, 32, 32)
    
#     x = rearrange(x, 'group b (nh nw) h w  ->group b (nh h) (nw w)',nh=8,nw=8)

#     x = x.unsqueeze(2)
    
#     return x

#   分段（均匀采样、直接分段）
def x_group(x, group=4):
    # x.shape = (b, 2048, 32, 32)
    # b = x.shape[0]
    b, t, h, w = x.shape
    # 均匀采样

    x= rearrange(x, 'b (ng group)  h w  ->group b ng h w',group=group) 
    # 直接分段
    # x= rearrange(x, 'b (group ng)  h w  ->group b ng h w',group=group)
    # Step 3: reshape to (d, b, 64, 8, 32, 32)
    x = x.view(group, b, 64, t//(group*64), 32, 32)
    # Step 4: mean across dimension 3 -> (d, b, 64, 1, 32, 32)
    x = x.mean(dim=3, keepdim=True)
    
    # Step 5: reshape to (d, b, 64, 32, 32)
    x = x.view(group, b, 64, 32, 32)
    x = group_replace_patch(x, ratio=0.25)
    x = rearrange(x, 'group b (nh nw) h w  ->group b (nh h) (nw w)',nh=8,nw=8)
    
    x = x.unsqueeze(2)
    
    return x


def x_group_rand(x, group=4):
    # x.shape = (b, 2048, 32, 32)
    # b = x.shape[0]
    b, t, h, w = x.shape
    # 均匀采样
    x= rearrange(x, 'b (ng group)  h w  ->group b ng h w',group=group) 
    # 直接分段
    # x= rearrange(x, 'b (group ng)  h w  ->group b ng h w',group=group)
    # Step 3: reshape to (d, b, 64, 8, 32, 32)
    x = x.view(group, b, 64, t//(group*64), 32, 32)
    # Step 4: mean across dimension 3 -> (d, b, 64, 1, 32, 32)
    x = x.mean(dim=3, keepdim=True)
    
    # Step 5: reshape to (d, b, 64, 32, 32)
    x = x.view(group, b, 64, 32, 32)
    
    x = rearrange(x, 'group b (nh nw) h w  ->group b (nh h) (nw w)',nh=8,nw=8)

    x = x.unsqueeze(2)
    
    return x


def to_patch(x, replace=False):
    # b 512 32 32
    b, t, h, w = x.shape
    x = time_norm(x, dim=1)
    groups = 64
    # b 512 32 32 -> b 64 8 32 32
    x = x.reshape(b, groups, t // groups, h, w)
    # b 64 8 32 32 -> b 64 1 32 32 -> b 64 32 32
    x = x.mean(dim=2).squeeze()

    if replace:
        x = replace_patch(x, ratio=1)
    # b 64 32 32 -> b 256 256 -> b 1 256 256
    x = rearrange(x, "b (th tw) h w -> b (th h) (tw w)", th=8, tw=8).unsqueeze(1)
    # x = x.repeat(1, 3, 1, 1)
    return x


def time_norm(x, dim=1, eps=1e-6):
    x = x - x.mean(dim=dim, keepdim=True)
    x = x / (x.std(dim=dim, keepdim=True) + eps)
    return x


if __name__ == "__main__":
    x = torch.randn(2, 512, 32, 32)
    x = replace_patch(x)
    print(x.shape)

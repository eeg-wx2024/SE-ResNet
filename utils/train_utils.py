import subprocess
import numpy as np
import torch
from datetime import datetime
import os
import argparse
import random
from datasets import EEGDataset
from models.maxvit import MaxvitNet
from models.resnet import resnet
import torch.nn as nn
from einops import rearrange
from torchvision.models.mnasnet import mnasnet0_5
from torchvision.models.maxvit import maxvit_t
from eeg_net import classifier_EEGChannelNet
from eeg_net import classifier_MLP
from models.simple import SimpleNet

class GroupResNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_model = resnet(attn_name="se", act_name="elu", blocks=[2, 2, 2, 2])

    def forward(self, x, y=None):
        group_num, batch_size =  x.shape[0], x.shape[1]
        x = rearrange(x, 'g b c h w ->(g  b) c h w')
        x = self.resnet_model(x)
        x = rearrange(x, '(g b) c-> g b c', g=group_num)

        
        if not self.training:
            return x.mean(dim=0)
        
        # 计算每个样本在每个组中概率最高的类别号
        _, max_indices = x.max(dim=2)  # shape: [5, 128]

        # 对每个样本进行投票，并找出对应的概率向量
        final_vectors = []
        
        for sample_idx in range(batch_size):  # 遍历每个样本
            # 获取当前样本的所有类别预测
            sample_votes = max_indices[:, sample_idx]
            
            # 计算最常出现的类别
            most_common_class= torch.bincount(sample_votes).argmax()

            # 找出这个类别首次出现的位置
            first_occurrence = (sample_votes == most_common_class).nonzero(as_tuple=True)[0][0]

            # 获取这个位置对应的概率向量
            selected_vector = x[first_occurrence, sample_idx]

            final_vectors.append(selected_vector)

        # 最终的概率向量列表
        final_vectors = torch.stack(final_vectors)
        # 返回每组中投票最多的类别的概率向量
        return final_vectors   





class GroupResNet_noscore(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_model = resnet(attn_name="se", act_name="elu", blocks=[2, 2, 2, 2])
        # elu,relu,silu
        # "se" "cbam""ca"

    def forward(self, x, y=None):
        group_num, batch_size =  x.shape[0], x.shape[1]
        
        # 将输入调整为 [group_num * batch_size, channels, height, width] 以适应 ResNet
        x = rearrange(x, 'g b c h w -> (g b) c h w')

        # 输入模型进行前向传播
        x = self.resnet_model(x)
        
        # 将输出重新调整为 [group_num, batch_size, channels] 的形状
        x = rearrange(x, '(g b) c -> g b c', g=group_num)
        
        if not self.training:
            return x.mean(dim=0)
        
        # 计算每个样本在每个组中概率最高的类别号
        _, max_indices = x.max(dim=2)  # shape: [group_num, batch_size]

        # 对每个样本进行投票，并找出对应的概率向量
        final_vectors = []
        
        for sample_idx in range(batch_size):  # 遍历每个样本
            # 获取当前样本的所有类别预测
            sample_votes = max_indices[:, sample_idx]
            
            # 计算最常出现的类别
            most_common_class = torch.bincount(sample_votes).argmax()

            # 找出这个类别首次出现的位置
            first_occurrence = (sample_votes == most_common_class).nonzero(as_tuple=True)[0][0]

            # 获取这个位置对应的概率向量
            selected_vector = x[first_occurrence, sample_idx]

            final_vectors.append(selected_vector)

        # 最终的概率向量列表
        final_vectors = torch.stack(final_vectors)
        # 返回每组中投票最多的类别的概率向量
        return final_vectors  


class GroupResNet(nn.Module):
    def __init__(self,group_num=1):
        super().__init__()

        # 要累计5个组的分数，初始分数为1000
        self.group_scores = torch.ones(group_num) * 1000

        self.resnet_model = resnet(attn_name="se", act_name="elu", blocks=[2, 2, 2, 2])


    def forward(self, x, y=None):
        device = x[0].device
      
        
        self.group_scores = self.group_scores.to(device)
        # x: 5, 128, 3, 224, 224
        group_num, batch_size =  x.shape[0], x.shape[1]
        x = rearrange(x, 'g b c h w ->(g  b) c h w')
        x = self.resnet_model(x)
        x = rearrange(x, '(g b) c-> g b c', g=group_num)

        
        if not self.training:
            return x.mean(dim=0)
        
        # 计算每个样本在每个组中概率最高的类别号
        _, max_indices = x.max(dim=2)  # shape: [5, 128]

        final_vectors = []
        

        # 初始化一个全零 tensor，用于累加每个类别的权重，大小为 [128, 40]
        # # 假设最多有 40 个类别
        batch_dict_group_indices = [{} for _ in range(batch_size)]
        # 遍历每个组
        for i in range(group_num):
            # 获取当前组的类别号
            group_indices = max_indices[i]  # shape: [128]
            # 获取当前组的权重
            group_weight = self.group_scores[i]
            # 累加当前组的权重到相应的类别号上
            for j in range(batch_size):
                if group_indices[j].cpu().numpy().item() in batch_dict_group_indices[j]:
                    batch_dict_group_indices[j][group_indices[j].cpu().numpy().item()] = group_weight+ batch_dict_group_indices[j][group_indices[j].cpu().numpy().item()]
                else:
                    batch_dict_group_indices[j][group_indices[j].cpu().numpy().item()] = group_weight

        # 对每个样本，找到权重最高的类别号
        top_category_indices = []
        for dict_group_indices in batch_dict_group_indices:
            max_value_key = max(dict_group_indices, key=dict_group_indices.get)  # 提取最大value对应的key
            top_category_indices.append(max_value_key)
   

        # 为每个样本找到对应的概率向量
        final_vectors = torch.zeros(batch_size, 40, device=device)
        for i in range(batch_size):
            category = top_category_indices[i]
            # 假设选择第一个组含有该类别的概率向量作为结果
            # 这里可以根据实际需求选择不同的策略
            # final_vectors[i] = stacked_outputs[0, i]
            for j in range(group_num):
                if max_indices[j, i] == category:
                    final_vectors[i] = x[j, i]
                    break

        # 检查 max_indices 中的每个类别号是否等于 y，并对相应的组进行加分
        if y is not None:
            for i in range(group_num):
                # matches = max_indices[i] == y  # 获取当前组中与 y 相匹配的情况
                matches = (max_indices[i] == y).int()*2-1
                # 猜错扣2分，猜对加2分
                matches[matches<0]=0
                matches[matches>0]=0.00001
                score_addition = matches.to(dtype=torch.float).sum()  # 计算需要加分的数量
                self.group_scores[i] += score_addition  # 对相应的组加分
                # print("group_scores:", self.group_scores)
                # if final_vectors.sum() == 0:
                    # print("final_vectors is 0")
                # 返回每个样本的最终概率向量
        return  final_vectors



def select_dataset(paths):
    return EEGDataset(paths)


def select_model():
    # model = GroupResNet()

    # model = SimpleNet()
    # model = MaxvitNet()
    
    model = GroupResNet_noscore()
    
    # model = classifier_EEGChannelNet()
    # model = classifier_MLP()

    # model = resnet(attn_name="se", act_name="elu", blocks=[2, 2, 2, 2])
    return model


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--N", type=int, default=1, help="N")
    parser.add_argument("--model", type=str, default="resnet", help="model name")
    args = parser.parse_args()
    args.pid = get_pid()
    return args


def get_log_dir():
    log_dir = "./log_visual/" + getNow() + "/"
    return log_dir + get_pid() + "/"


def get_pid():
    pid = str(os.getpid())[-3:]
    return pid


def getNow():
    now = datetime.now()
    current_year = now.year % 100
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    current_minute = now.minute
    return (
        str(current_year).zfill(2)
        + "/"
        + str(current_month).zfill(2)
        + "/"
        + str(current_day).zfill(2)
        + "/"
        + str(current_hour).zfill(2)
        + str(current_minute).zfill(2)
    )


def get_device(gpus):
    device = torch.device(
        f"cuda:{get_gpu_usage(gpus)}" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)
    return device


def get_gpu_usage(gpus):
    """Returns a list of dictionaries containing GPU usage information."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
        ],
        encoding="utf-8",
    )
    lines = output.strip().split("\n")
    # 分离已用内存和总内存，转换为 numpy 数组
    memory = np.array([line.strip().split(",") for line in lines], dtype=int)

    # 计算内存使用百分比
    memory_used_percentage = ((memory[:, 0] / memory[:, 1]) * 100).astype(int)

    # 更新不在 only_use 中的 GPU 的使用率为 100%

    if gpus is not None:
        mask = np.ones(len(memory_used_percentage), dtype=bool)
        mask[gpus] = False
        memory_used_percentage[mask] = 100

    print(memory_used_percentage)
    # 返回最小内存使用率的 GPU 索引
    return np.argmin(memory_used_percentage)


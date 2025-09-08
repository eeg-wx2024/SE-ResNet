import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import mnasnet0_5


class SimpleNet(nn.Module):
    
    def __init__(self,group_num=4):
        super().__init__()

        # 要累计5个组的分数，初始分数为1000
        self.group_scores = torch.ones(group_num) * 1000

        self.model = mnasnet0_5(num_classes = 50)
        


    def forward(self, x, y=None):
        device = x[0].device
      
        
        self.group_scores = self.group_scores.to(device)
        # x: 5, 128, 3, 224, 224
        group_num, batch_size =  x.shape[0], x.shape[1]
        x = rearrange(x, 'g b c h w ->(g  b) c h w')
        x = self.model(x)
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
                matches[matches>0]=1
                score_addition = matches.to(dtype=torch.float).sum()  # 计算需要加分的数量
                self.group_scores[i] += score_addition  # 对相应的组加分
        # print("group_scores:", self.group_scores)
        # if final_vectors.sum() == 0:
        #     print("final_vectors is 0")
        # 返回每个样本的最终概率向量
            return  final_vectors

if __name__ == "__main__":
    x = torch.rand(5, 128, 3, 224, 224)
    model = SimpleNet()
    print(model(x).shape)
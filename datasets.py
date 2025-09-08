from torch.utils.data import Dataset, Sampler
import pickle
from collections import defaultdict
import torch
import numpy as np


class EEGDataset(Dataset):
    def __init__(self, paths):
        self.filepaths = paths
        self.labels = [
            int(fp.split("_")[-2].replace(".pkl", "")) for fp in self.filepaths
        ]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        return self.load_data(self.filepaths[idx])

    def load_data(self, filepath):
        with open(filepath, "rb") as f:
            # 512, 32, 32
            x = torch.tensor(pickle.load(f), dtype=torch.float)
            y = torch.tensor(pickle.load(f), dtype=torch.long)
            try:
                assert 0 <= y <= 39
            except AssertionError:
                print(f"Error: {filepath}")
        return x, y


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, N=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples_per_class = N
        self.labels = dataset.labels
        self.num_batches = len(self.labels) // (
            self.batch_size * self.n_samples_per_class
        )
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        self.label_keys = list(self.label_to_indices.keys())

        repeats = (self.batch_size // len(self.label_keys)) + 1
        self.extended_label_keys = self.label_keys * repeats  # 复制label_keys

    def __iter__(self):

        # 用于存储所有批次的数组
        batches = np.empty(
            (self.num_batches, self.batch_size * self.n_samples_per_class), dtype=int
        )

        for i in range(self.num_batches):
            batch = np.empty((0,), dtype=int)
            classes = np.random.choice(
                self.extended_label_keys, self.batch_size, replace=False
            )

            for class_ in classes:
                indices = np.random.choice(
                    self.label_to_indices[class_],
                    self.n_samples_per_class,
                    replace=False,
                )
                batch = np.append(batch, indices)
            batches[i, :] = batch

        # 随机化批次
        np.random.shuffle(batches)

        for batch in batches:
            yield batch.tolist()  # DataLoader期望Python列表作为输出

    def __len__(self):
        return len(self.dataset) // (self.batch_size * self.n_samples_per_class)


# 实现一个转换函数，计算同一类别中N个样本的平均值
def collate_fn(batch, N=1):
    # 使用zip和星号操作符解压batch，直接转换为张量
    data_list, label_list = zip(*batch)  # 这将返回两个元组，分别包含所有数据和所有标签

    # 直接将列表转换为堆叠的张量
    data_tensor = torch.stack(data_list)
    labels_tensor = torch.tensor(label_list)

    # 假设每类样本数量N为10
    data_tensor = data_tensor.view(-1, N, *data_tensor.shape[1:])
    data_mean = data_tensor.mean(dim=1)

    labels_tensor = labels_tensor.view(-1, N)
    labels_mean = labels_tensor[:, 0]  # 选取每个批次中第一个标签作为代表标签

    return data_mean, labels_mean

import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import torch.nn.functional as F
import os
import mne
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import x_group
from utils.train_utils import (
    fix_random_seed,
    get_args,
    get_device,
    get_log_dir,
    select_model,
)

# 固定随机种子
seed = 1234
fix_random_seed(seed)

args = get_args()
print(args)

# 设备
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
device = get_device(gpus)

k_fold = KFold(n_splits=5, shuffle=True)

# tensorboard
tb = SummaryWriter(log_dir=get_log_dir())

def normalize_samples(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    x_normalized = (x - mean) / (std + 1e-6)
    return x_normalized

class MNEReader(object):
    def __init__(self, filetype='edf', method='manual', length=500, exclude=(), stim_channel='auto'):
        self.filetype = filetype
        self.length = length
        self.exclude = exclude
        self.stim_channel = stim_channel
        self.method = self.read_by_manual
        self.file_path = None
        self.set = None

    def get_set(self, file_path, stim_list):
        self.file_path = file_path
        self.set = self.method(stim_list)
        return self.set

    def read_raw(self):
        if self.filetype == 'edf':
            raw = mne.io.read_raw_edf(self.file_path, preload=True, exclude=self.exclude, stim_channel=self.stim_channel)
        else:
            raise Exception('Unsupported file type!')
        return raw

    def read_by_manual(self, stim_list):
        raw = self.read_raw()
        picks = mne.pick_types(raw.info, eeg=True, stim=False)
        set = []
        for i in stim_list:
            end = i + self.length
            data, _ = raw[picks, i:end]
            set.append(data.T)
        return set

def ziyan_read(file_path):
    with open(file_path) as f:
        stim = []
        target_class = []
        for line in f.readlines():
            if line.strip().startswith('Stimulus'):
                line = line.strip().split(',')
                classes = int(line[1][-2:])
                time = int(line[2].strip())
                stim.append(time)
                target_class.append(classes)
    return stim, target_class

def find_edf_and_markers_files(base_path, file_prefix=None):
    edf_files = {}
    for filename in os.listdir(base_path):
        if filename.endswith('.edf') and (file_prefix is None or filename.startswith(file_prefix)):
            base_name = filename[:-4]
            edf_files[base_name] = {
                'edf': os.path.join(base_path, filename),
                'markers': os.path.join(base_path, base_name + '.Markers')
            }
    return edf_files

def load_and_preprocess_data(edf_file_path, label_file_path):
    edf_reader = MNEReader(filetype='edf', method='manual', length=500)
    stim, target_class = ziyan_read(label_file_path)

    # 将标签值减1，以使标签范围从0到49
    target_class = [cls - 1 for cls in target_class]

    xx = edf_reader.get_set(file_path=edf_file_path, stim_list=stim)

    xx_np = np.array(xx)

    # 如果通道数不是127，跳过
    if xx_np.shape[2] != 127:
        return None, None

    xx_normalized = normalize_samples(xx_np)

    eeg_data = np.transpose(xx_normalized, (0, 2, 1))
    eeg_data = eeg_data[:, np.newaxis, :, :]  # 添加一个维度

    eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    labels_tensor = torch.tensor(target_class, dtype=torch.long)
    
    return eeg_data_tensor, labels_tensor

# 主代码部分
base_path = '/data1/wuxia/dataset/Face_EEG'
edf_files = find_edf_and_markers_files(base_path)

all_eeg_data = []
all_labels = []

for base_name, files in edf_files.items():
    edf_file_path = files['edf']
    label_file_path = files['markers']

    if not os.path.exists(label_file_path):
        logging.info(f"Markers file for {edf_file_path} does not exist. Skipping.")
        continue

    eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path)
    
    if eeg_data is None or labels is None:
        continue
    
    all_eeg_data.append(eeg_data)
    all_labels.append(labels)

if len(all_eeg_data) == 0:
    logging.info("No valid EEG data found.")
    exit()

# 将所有数据拼接为一个整体
all_eeg_data = torch.cat(all_eeg_data)
all_labels = torch.cat(all_labels)

# K-Fold 交叉验证
for fold, (train_idx, val_idx) in enumerate(k_fold.split(all_eeg_data)):
    train_dataset = TensorDataset(all_eeg_data[train_idx], all_labels[train_idx])
    val_dataset = TensorDataset(all_eeg_data[val_idx], all_labels[val_idx])

    # 模型
    model = select_model().to(device)
    if fold == 0:
        print("num_params:", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e4, "w")
        print(model)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        # 训练
        model.train()
        total_correct = 0
        total_sample = 0
        total_loss = 0
        for step, (x, y) in enumerate(train_loader):
            total_sample += x.shape[0]
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            # x = x_group(x)
            print(x.shape)
            y_pred = model(x, y)
            print(y_pred.shape)
            print(y.shape)
            loss_cls = F.cross_entropy(y_pred, y)
            loss = loss_cls

            total_loss += loss.item()
            total_correct += (y_pred.argmax(dim=1) == y).sum().item()

            loss.backward()
            optimizer.step()

            if step % (len(train_loader) // 5) == 0:
                tb.add_scalar(f"fold {fold}/train/loss_cls", loss_cls.item(), epoch * len(train_loader) + step)

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_sample * 100
        tb.add_scalar(f"fold {fold}/train/loss", train_loss, epoch)
        tb.add_scalar(f"fold {fold}/train/acc", train_acc, epoch)

        # 验证
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total_sample = 0
            for x, y in val_loader:
                total_sample += x.shape[0]
                x = x.to(device)
                y = y.to(device)
                # x = x_group(x)
                y_pred = model(x)
                loss_cls = F.cross_entropy(y_pred, y)
                total_loss += loss_cls.item()
                total_correct += (y_pred.argmax(dim=1) == y).sum().item()

            val_loss = total_loss / len(val_loader)
            val_acc = total_correct / total_sample * 100
            tb.add_scalar(f"fold {fold}/val/loss", val_loss, epoch)
            tb.add_scalar(f"fold {fold}/val/acc", val_acc, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
            print(f"fold {fold}, train_loss {train_loss:.4f}, train_acc {train_acc:.2f}, val_loss {val_loss:.4f}, val_acc {val_acc:.2f}, best_val_acc {best_val_acc:.2f} (epoch {best_epoch}/{epoch})")

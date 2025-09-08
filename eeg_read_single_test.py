import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
from sklearn.model_selection import KFold

def normalize_samples(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    x_normalized = (x - mean) / (std + 1e-6)
    return x_normalized

class MNEReader(object):
    def __init__(self, filetype='edf', method='stim', resample=None, length=500, exclude=(), stim_channel='auto', montage=None):
        self.filetype = filetype
        self.file_path = None
        self.resample = resample
        self.length = length
        self.exclude = exclude
        self.stim_channel = stim_channel
        self.montage = montage
        if stim_channel == 'auto':
            assert method == 'manual'

        if method == 'auto':
            self.method = self.read_auto
        elif method == 'stim':
            self.method = self.read_by_stim
        elif method == 'manual':
            self.method = self.read_by_manual
        self.set = None
        self.pos = None

    def get_set(self, file_path, stim_list=None):
        self.file_path = file_path
        self.set = self.method(stim_list)
        return self.set

    def get_pos(self):
        assert self.set is not None
        return self.pos

    def get_item(self, file_path, sample_idx, stim_list=None):
        if self.file_path == file_path:
            return self.set[sample_idx]
        else:
            self.file_path = file_path
            self.set = self.method(stim_list)
            return self.set[sample_idx]

    def read_raw(self):
        if self.filetype == 'bdf':
            raw = mne.io.read_raw_bdf(self.file_path, preload=True, exclude=self.exclude, stim_channel=self.stim_channel)
            print(raw.info['sfreq'])
        elif self.filetype == 'edf':
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
            data, times = raw[picks, i:end]
            set.append(data.T)
        return set

    def read_auto(self, *args):
        raw = self.read_raw()
        events = mne.find_events(raw, stim_channel=self.stim_channel, initial_event=True, output='step')
        event_dict = {'stim': 65281, 'end': 0}
        epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True).drop_channels('Status')
        epochs.equalize_event_counts(['stim'])
        stim_epochs = epochs['stim']
        del raw, epochs, events
        return stim_epochs.get_data().transpose(0, 2, 1)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.max_norm = kwargs.pop('max_norm', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm:
            self.weight.data = torch.renorm(self.weight.data, 2, 0, self.max_norm)
        return super().forward(x)

class EEGNet(nn.Module):
    def __init__(self, F1=8, F2=16, D=2, K1=64, K2=16, n_timesteps=500, n_electrodes=127, n_classes=50, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(1, F1, (1, K1), padding=(0, K1 // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (n_electrodes, 1), bias=False, groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        self.act1 = nn.ELU()

        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, K2), bias=False, groups=F1 * D, padding=(0, K2 // 2))
        self.conv4 = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)

        self.act2 = nn.ELU()

        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(F2 * (n_timesteps // 32), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act2(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x

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

def find_edf_and_markers_files(base_path, file_prefix):
    edf_files = {}
    for filename in os.listdir(base_path):
        if filename.startswith(file_prefix) and filename.endswith('.edf'):
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
    print(f"{os.path.basename(edf_file_path)} - xx_np.shape=", xx_np.shape)

    # 如果通道数不是127，跳过
    if xx_np.shape[2] != 127:
        print(f"Skipping file {edf_file_path}, expected 127 channels but got {xx_np.shape[2]}.")
        return None, None

    xx_normalized = normalize_samples(xx_np)
    print(f"{os.path.basename(edf_file_path)} - xx_normalized.shape=", xx_normalized.shape)

    eeg_data = np.transpose(xx_normalized, (0, 2, 1))
    eeg_data = eeg_data[:, np.newaxis, :, :]
    print(f"{os.path.basename(edf_file_path)} - eeg_data.shape=", eeg_data.shape)

    eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    labels_tensor = torch.tensor(target_class, dtype=torch.long)
    
    return eeg_data_tensor, labels_tensor

def main():
    base_path = '/data1/wuxia/dataset/Face_EEG'
    file_prefix = 'aaa'  # 修改为你需要的文件前缀
    edf_files = find_edf_and_markers_files(base_path, file_prefix)

    all_eeg_data = []
    all_labels = []
    invalid_files = []

    for base_name, files in edf_files.items():
        edf_file_path = files['edf']
        label_file_path = files['markers']

        if not os.path.exists(label_file_path):
            print(f"Markers file for {edf_file_path} does not exist. Skipping.")
            continue

        eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path)
        
        if eeg_data is None or labels is None:
            invalid_files.append(edf_file_path)
            continue
        
        all_eeg_data.append(eeg_data)
        all_labels.append(labels)

    all_eeg_data = torch.cat(all_eeg_data)
    all_labels = torch.cat(all_labels)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    num_epochs = 300

    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_eeg_data)):
        print(f"FOLD {fold+1}")

        train_dataset = TensorDataset(all_eeg_data[train_idx], all_labels[train_idx])
        test_dataset = TensorDataset(all_eeg_data[test_idx], all_labels[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = EEGNet(n_timesteps=500, n_electrodes=127, n_classes=50)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total

            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            test_acc = 100 * correct_test / total_test

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')

    if invalid_files:
        print("Files skipped due to invalid channel size:")
        for invalid_file in invalid_files:
            print(invalid_file)

if __name__ == '__main__':
    main()

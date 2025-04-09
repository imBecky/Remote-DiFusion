import scipy.io as sio
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    ============= Dataset Class implementation =============
    """
    def __init__(self, data_dict, labels, transform=None):
        self.data_dict = data_dict
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """根据索引返回单个样本（数据和标签）"""
        data_dict = {'rgb': self.data_dict['rgb'][idx], 'hsi': self.data_dict['hsi'][idx],
                     'lidar': self.data_dict['lidar'][idx]}
        label = self.labels[idx]

        if self.transform:
            data_dict['rgb'] = self.transform(data_dict['rgb'])
            data_dict['hsi'] = self.transform(data_dict['hsi'])
            data_dict['lidar'] = self.transform(data_dict['lidar'])

        return data_dict, label


def data_report(dataset_name):
    if dataset_name == "SZU_R1":
        print(f'Data Report of SZU Tree Dataset R1:\n'
              f'RGB data shape: torch.Size([3, 4810, 6170])\n'
              f'HSI data shape: torch.Size([98, 2405, 3085])\n'
              f'lidar data shape: torch.Size([1, 2405, 3085])\n'
              f'labels shape: {labels.shape}')
    else:
        pass


"""
========= loading SZU dataset ========
hsi and lidar data are using matlab 7.3+, thus load them via h5py.
rgb data and labels can be loaded simply via sio.loadmat()
-> SZUTree_Dataset_R1, SZUTree_R1_dataloader
"""
file_path_RGB = './dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_RGB_R1/SZUTreeRGB_R1.mat'
file_path_HSI = './dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_HSI_R1/data_band98.mat'
file_path_CHM = './dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_CHM_R1/SZUTreeCHM_R1.mat'

szu_data_dict = {}
for i, file_path in enumerate([file_path_HSI, file_path_CHM]):
    with h5py.File(file_path, 'r') as file:
        key = list(file.keys())
        data = file[key[0]][()]
        data = data.T
        print(data.shape)
        if i == 0:
            transposed_data = torch.from_numpy(np.transpose(data, (2, 0, 1)))
        else:
            transposed_data = torch.from_numpy(data).unsqueeze(0)
        szu_data_dict['hsi' if i == 0 else 'lidar'] = transposed_data

rgb = sio.loadmat(file_path_RGB)['data']
szu_data_dict['rgb'] = torch.from_numpy(rgb)
labels = sio.loadmat('./dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/Annotations_SZUTreeData_R1'
                     '/SZUTreeData_R1_typeid_with_labels_5cm.mat')
labels = torch.from_numpy(labels['data'])
SZUTree_Dataset_R1_raw = CustomDataset(data_dict=szu_data_dict, labels=labels)


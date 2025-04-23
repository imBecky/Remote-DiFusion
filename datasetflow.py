import scipy.io as sio
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def align_modalities(data_dict):
    """将HSI/LiDAR上采样至RGB分辨率"""
    # RGB尺寸 (H, W) = (4810, 6170)
    # HSI/LiDAR尺寸 (H, W) = (2405, 3085)

    # 双线性插值上采样2倍
    data_dict['hsi'] = F.interpolate(data_dict['hsi'].unsqueeze(0).float(),
                                     size=(4810, 6170),
                                     mode='bilinear')[0]  # -> [98, 4810, 6170]

    data_dict['lidar'] = F.interpolate(data_dict['lidar'].unsqueeze(0).float(),
                                       size=(4810, 6170),
                                       mode='bilinear')[0]  # -> [1, 4810, 6170]
    return data_dict


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
              f'HSI data shape: torch.Size([98, 4810, 6170])\n'
              f'lidar data shape: torch.Size([1, 4810, 6170])\n'
              f'labels shape: [4810, 6170]')
    else:
        pass


"""
========= loading SZU dataset ========
hsi and lidar data are using matlab 7.3+, thus load them via h5py.
rgb data and labels can be loaded simply via sio.loadmat()
-> SZUTree_Dataset_R1, SZUTree_R1_dataloader
"""


def upsample_and_save():
    file_path_RGB = './dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_RGB_R1/SZUTreeRGB_R1.mat'
    file_path_HSI = './dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_HSI_R1/data_band98.mat'
    file_path_CHM = './dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_CHM_R1/SZUTreeCHM_R1.mat'

    szu_data_dict = {}
    for i, file_path in enumerate([file_path_HSI, file_path_CHM]):
        with h5py.File(file_path, 'r') as file:
            key = list(file.keys())
            data = file[key[0]][()]
            data = data.T
            if i == 0:
                transposed_data = torch.from_numpy(np.transpose(data, (2, 0, 1)))
            else:
                transposed_data = torch.from_numpy(data).unsqueeze(0)
            szu_data_dict['hsi' if i == 0 else 'lidar'] = transposed_data

    rgb = sio.loadmat(file_path_RGB)['data']
    szu_data_dict['rgb'] = torch.from_numpy(rgb)
    SZUTreeDataset_Upsampled = align_modalities(szu_data_dict)
    torch.save(SZUTreeDataset_Upsampled, 'SZUTreeDataset_Upsampled.pt')


def get_SZUTree_R1_dataset(dataset_path, label_path):
    data_dict = torch.load(dataset_path, weights_only=True)
    data_dict['hsi'] = data_dict['hsi'].to(torch.uint16).to(CUDA0)
    data_dict['lidar'] = data_dict['lidar'].to(torch.uint16).to(CUDA0)
    data_dict['rgb'] = data_dict['rgb'].to(CUDA0)
    labels = sio.loadmat(label_path)
    labels = torch.from_numpy(labels['data']).to(CUDA0)

    SZUTree_Dataset_R1 = CustomDataset(data_dict=data_dict, labels=labels)
    return SZUTree_Dataset_R1


dataset_path = './dataset/SZUTreeData2.0/SZUTreeDataset_Upsampled.pt'
label_path = './dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/Annotations_SZUTreeData_R1' \
             '/SZUTreeData_R1_typeid_with_labels_5cm.mat '
SZUTree_Dataset_R1 = get_SZUTree_R1_dataset(dataset_path, label_path)

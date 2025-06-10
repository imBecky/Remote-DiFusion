import scipy.io as sio
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset, random_split, DataLoader
import pytorch_lightning as pl

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

    def __init__(self, data_dict, label, patch_size=256, stride=128, transform=None):
        self.data_dict = data_dict
        self.rgb = data_dict['rgb']
        self.hsi = data_dict['hsi']
        self.lidar = data_dict['lidar']
        self.label = label
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        # calculate patches number
        self.height = self.data_dict['rgb'].shape[1]
        self.width = self.data_dict['rgb'].shape[2]
        self.num_patches_h = (self.height - patch_size) // stride + 1
        self.num_patches_w = (self.width - patch_size) // stride + 1
        self.num_patches = self.num_patches_h * self.num_patches_w

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        """根据索引返回单个样本（数据和标签）"""

        h_idx = idx // self.num_patches_w  # 行索引
        w_idx = idx % self.num_patches_w  # 列索引
        # 计算patch的起始位置
        h_start = h_idx * self.stride
        w_start = w_idx * self.stride

        if self.transform:
            self.rgb = self.transform(self.rgb).to(CUDA0)
            self.hsi = self.transform(self.hsi).to(CUDA0)
            self.lidar = self.transform(self.lidar).to(CUDA0)

        rgb_patch = self.rgb[:, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
        hsi_patch = self.hsi[:, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
        lidar_patch = self.lidar[:, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
        label_patch = self.label[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size].to(CUDA0)

        data_dict = {'rgb': rgb_patch,
                     'hsi': hsi_patch,
                     'lidar': lidar_patch}
        return data_dict, label_patch


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
    file_path_RGB = '../autodl-fs/dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_RGB_R1/SZUTreeRGB_R1.mat'
    file_path_HSI = '../autodl-fs/dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_HSI_R1/data_band98.mat'
    file_path_CHM = '../autodl-fs/dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_CHM_R1/SZUTreeCHM_R1.mat'

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
    # torch.save(SZUTreeDataset_Upsampled, 'SZUTreeDataset_Upsampled.pt')
    return SZUTreeDataset_Upsampled


def get_SZUTree_R1_dataset(Dataset, label_path):
    data_dict = Dataset
    data_dict['hsi'] = data_dict['hsi'].to(torch.uint16).to(CUDA0)
    data_dict['lidar'] = data_dict['lidar'].to(torch.uint16).to(CUDA0)
    data_dict['rgb'] = data_dict['rgb'].to(CUDA0)
    label = sio.loadmat(label_path)
    label = torch.from_numpy(label['data']).to(CUDA0)

    Dataset_R1 = CustomDataset(data_dict=data_dict, label=label)
    return Dataset_R1


dataset_path = '../autodl-fs/dataset/SZUTreeData2.0/SZUTreeDataset_Upsampled.pt'
label_path = '../autodl-fs/dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/Annotations_SZUTreeData_R1' \
             '/SZUTreeData_R1_typeid_with_labels_5cm'
SZUTreeDataset_Upsampled = upsample_and_save()
SZUTree_Dataset_R1 = get_SZUTree_R1_dataset(SZUTreeDataset_Upsampled, label_path)
SZUTree_Dataset_R1_subset = Subset(SZUTree_Dataset_R1, indices=range(256))
using_dataset = SZUTree_Dataset_R1
train_size = int(0.6 * len(using_dataset))
val_size = int(0.2 * len(using_dataset))
test_size = len(using_dataset) - train_size - val_size

SZUTree_train_set, SZUTree_val_set, SZUTree_test_set = random_split(
    using_dataset, [train_size, val_size, test_size],
    generator=torch.Generator()
)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_set, val_set, test_set, bs):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.bs = bs

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, shuffle=True, drop_last=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, shuffle=False, drop_last=True, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, shuffle=False, drop_last=True, num_workers=0)


def get_data_module(name, bs):
    if name == "SZU_R1":
        data_module = MyDataModule(SZUTree_train_set, SZUTree_val_set, SZUTree_test_set, bs)
        return data_module

import scipy.io as sio
# import h5py
# import numpy as np

# with h5py.File('./dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_HSI_R1/data_band98.mat', 'r') as f:
#     # data = np.array(f['data'][:])
#     print(f)
    # data = data.T  # 根据实际数据调整
#
szu_tree_r1_vhr = sio.loadmat('./dataset/SZUTreeData2.0/SZUTreeData_R1_2.0/SZUTree_RGB_R1/SZUTreeRGB_R1.mat')
# for i, data in enumerate([szu_tree_r1_vhr, szu_tree_r1_hsi, szu_tree_lidar]):
#     print(data.shape)

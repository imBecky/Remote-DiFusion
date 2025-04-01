import torch
import numpy as np


def data_report(root):
    hsi = torch.load(root + 'hsi.pth', weights_only=False)
    ndsm = torch.load(root + 'ndsm.pth', weights_only=False)
    rgb = torch.load(root + 'rgb.pth', weights_only=False)
    gt = torch.load(root + 'gt.pth', weights_only=False)
    print(f'hsi shape:{hsi.shape}')
    print(f'ndsm shape:{ndsm.shape}')
    print(f'rgb shape:{rgb.shape}')
    print(f'gt shape:{gt.shape}')

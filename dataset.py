import torch


def data_report(root):
    hsi = torch.load(root + 'small/' + 'hsi.pth', weights_only=False)
    ndsm = torch.load(root + 'small/' + 'ndsm.pth', weights_only=False)
    rgb = torch.load(root + 'small/' + 'rgb.pth', weights_only=False)
    gt = torch.load(root + 'small/' + 'gt.pth', weights_only=False)
    print(f'hsi shape:{hsi.shape}')
    print(f'ndsm shape:{ndsm.shape}')
    print(f'rgb shape:{rgb.shape}')
    print(f'gt shape:{gt.shape}')

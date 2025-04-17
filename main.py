import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch import optim
from datasetflow import data_report, SZUTree_Dataset_R1
from Model_old import ResNet50Encoder
from utils import set_seed, Dataset_from_feature, SplitDataset, CosineSimilarityLoss

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_argument_parse():
    parse = argparse.ArgumentParser(description='parameters')
    parse.add_argument('--trial_run', type=int, default=0, help='Experiment number of the trial run.')
    parse.add_argument('--dataset', type=str, default='Houston', help='DFC Houston 2018 | SZUTreeData')
    parse.add_argument('--lr1', type=float, default=0.0005, help='learning rate for noise predictor')
    parse.add_argument('--lr2', type=float, default=0.0005, help='learning rate for classifier')
    parse.add_argument('--lr3', type=float, default=0.0005, help='learning rate for GAN block')
    parse.add_argument('--bs', type=int, default=30, help='batch_size')
    parse.add_argument('--seed', type=int, default=13, help='default seed = 13')
    parse.add_argument('--T', type=int, default=1000, help='time steps for diffusion procedure')
    parse.add_argument('--if_small_dataset', type=int, default=1)
    parse.add_argument('--epoch', type=int, default=30)
    parse.add_argument('--dataset_ratio', type=float, default=0.8, help='Split ratio of train and test dataset')
    parse.add_argument('--image_size', type=int, default=32, help="")  # TODO: complete the help
    parse.add_argument('--feature_channels', type=int, default=1, help="")  # TODO: complete the help
    parse.add_argument('--dim_mults', type=tuple, default=(1, 2, 4), help='dims of ?')  # TODO: complete the help
    parse.add_argument('--dr', type=float, default=0.5, help='dropout rate of classifier')
    parse.add_argument('--log_dir', default='./logs')
    args = parse.parse_args()
    return args


def trian(args):
    # setting init
    torch.set_float32_matmul_precision('medium')
    set_seed(args.seed)
    if args.trial_run == 0:
        """Data Report of small dataset"""
        data_report(args.dataset)
    elif args.trial_run == 1:
        """ Encode modals, generate modal-invariant features"""

    else:
        print("\033[91m Trial Run设置错误！！！\033[0m")


if __name__ == '__main__':
    args = get_argument_parse()
    trian(args)

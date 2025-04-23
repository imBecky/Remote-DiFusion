import argparse
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch import optim
from datasetflow import data_report, SZUTree_Dataset_R1, SZUTree_Dataset_R1_subset
from Model_old import ResNet50Encoder
from utils import set_seed, Dataset_from_feature, SplitDataset, CosineSimilarityLoss
from model_base import InvariantGenerator

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_argument_parse():
    parse = argparse.ArgumentParser(description='parameters')
    parse.add_argument('--trial_run', type=int, default=0, help='Experiment number of the trial run.')
    parse.add_argument('--dataset', type=str, default='SZU_R1', help='DFC Houston 2018 | SZUTreeData')
    parse.add_argument('--lr1', type=float, default=0.0005, help='learning rate for noise predictor')
    parse.add_argument('--lr2', type=float, default=0.0005, help='learning rate for classifier')
    parse.add_argument('--lr3', type=float, default=0.0005, help='learning rate for GAN block')
    parse.add_argument('--bs', type=int, default=16, help='batch_size')
    parse.add_argument('--seed', type=int, default=13, help='default seed = 13')
    parse.add_argument('--T', type=int, default=1000, help='time steps for diffusion procedure')
    parse.add_argument('--if_small_dataset', type=int, default=1)
    parse.add_argument('--epoch', type=int, default=30)
    parse.add_argument('--dataset_ratio', type=float, default=0.8, help='Split ratio of train and test dataset')
    parse.add_argument('--image_size', type=int, default=32, help="")  # TODO: complete the help
    parse.add_argument('--feature_channels', type=int, default=1, help="")  # TODO: complete the help
    parse.add_argument('--dim_mults', type=tuple, default=(1, 2, 4), help='dims of ?')  # TODO: complete the help
    parse.add_argument('--dr', type=float, default=0.5, help='dropout rate of classifier')
    parse.add_argument('--betas', type=str, default='0.5,0.999',
                        help='Adam optimizer betas parameters (beta1,beta2). Default: 0.5,0.999')
    parse.add_argument('--log_dir', default='./logs')
    args = parse.parse_args()
    # 转换 betas 字符串为浮点数元组
    try:
        args.betas = tuple(map(float, args.betas.split(',')))
        if len(args.betas) != 2:
            raise ValueError
    except:
        raise argparse.ArgumentTypeError(
            "betas must be two comma-separated floats (e.g. '0.5,0.999')")
    return args


def train(args):
    # setting init
    torch.set_float32_matmul_precision('medium')
    set_seed(args.seed)
    checkpointer = ModelCheckpoint(dirpath=f"logs/trial{args.trial_run}/",
                                   filename='latest',
                                   monitor="cls_acc",
                                   save_last=True,
                                   save_weights_only=False,
                                   mode='max')
    logger = TensorBoardLogger("logs", name=f'Trial{args.trial_run}-{args.dataset}')
    if args.trial_run == 0:
        """Data Report of small dataset"""
        data_report(args.dataset)
    elif args.trial_run == 1:
        """ Encode modals, generate modal-invariant features"""
        print("============== Running[ Trial 1 ] ==============")
        if args.dataset == "SZU_R1":
            data_loader = DataLoader(SZUTree_Dataset_R1_subset, batch_size=args.bs, shuffle=True)
            trainer = Trainer(accelerator='gpu',
                              devices='auto',
                              max_epochs=args.epoch,
                              callbacks=[checkpointer],
                              logger=logger)
            module = InvariantGenerator(args)
            trainer.fit(module, data_loader)
    else:
        print("\033[91m Trial Run设置错误！！！\033[0m")


if __name__ == '__main__':
    args = get_argument_parse()
    args.trial_run = 1
    # print(args)
    train(args)

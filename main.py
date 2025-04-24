import argparse
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from datasetflow import data_report, get_data_module
from config import args
from utils import set_seed
from model_base import InvariantGenerator

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            data_module = get_data_module(args.dataset, args.bs)
            trainer = Trainer(accelerator='gpu',
                              devices='auto',
                              max_epochs=args.epoch,
                              callbacks=[checkpointer],
                              logger=logger)
            module = InvariantGenerator(args)
            trainer.fit(module, data_module)
    else:
        print("\033[91m Trial Run设置错误！！！\033[0m")


if __name__ == '__main__':
    args = args
    # args.trial_run = 1
    # print(args)
    train(args)

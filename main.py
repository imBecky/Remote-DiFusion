import argparse
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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
    checkpointer = ModelCheckpoint(dirpath=f"logs/trial{args.trial_run}/"+'model-{epoch:02d}-{val_cls_acc:.2f}',
                                   filename='latest',
                                   monitor="val_cls_acc",
                                   save_last=True,
                                   save_weights_only=False,
                                   mode='max',
                                   save_top_k=True)
    early_stop_callback = EarlyStopping(
        monitor='val_cls_acc',
        patience=10,  # 10个epoch没有改善就停止
        mode='max'
    )
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
                              callbacks=[checkpointer, early_stop_callback],
                              deterministic=True,
                              logger=logger,
                              log_every_n_steps=1,  # 每1步记录一次日志 TODO: small batch training
                              check_val_every_n_epoch=1  # 每1个epoch验证一次
                              )
            module = InvariantGenerator(args)
            trainer.fit(module, datamodule=data_module)
            trainer.test(ckpt_path="best", datamodule=data_module)
    else:
        print("\033[91m Trial Run设置错误！！！\033[0m")


if __name__ == '__main__':
    args = args
    args.trial_run = 1
    # print(args)
    train(args)

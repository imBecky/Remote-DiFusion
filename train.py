import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import pytorch_lightning as pl
from torch import optim
from datasetflow import data_report, SZUTree_Dataset_R1_raw
from Model import ResNet50Encoder
from utils import set_seed, Dataset_from_feature, SplitDataset, CosineSimilarityLoss

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_argument_parse():
    parse = argparse.ArgumentParser(description='parameters')
    parse.add_argument('--trial_run', type=int, default=0, help='Experiment number of the trial run.')
    parse.add_argument('--dataset', type=str, default='Houston', help='DFC Houston 2018 | SZUTreeData')
    parse.add_argument('--lr1', type=float, default=0.0005, help='learning rate for noise predictor')
    parse.add_argument('--lr2', type=float, default=0.0005, help='learning rate for classifier')
    parse.add_argument('--lr3', type=float, default=0.005, help='learning rate for discriminator')
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


class Trainer(nn.Module):
    def __init__(self, models, args):
        super(Trainer, self).__init__()
        self.args = args
        self.models = models
        self.denoiser_criterion = None
        self.classifier_criterion = None
        self.discriminator_criterion = None
        self.denoiser_optimizer_vhr = None
        self.denoiser_optimizer_lidar = None
        self.denoiser_optimizer_hsi = None
        self.classifier_optimizer = None
        self.discriminator_optimizer = None

        self.data_loader_train = None
        self.data_loader_test = None

    def init_denoiser(self, loss=F.smooth_l1_loss, optimizer=optim.Adam):
        self.models.init_denoisers()
        self.denoiser_criterion = loss
        self.denoiser_optimizer_vhr = optimizer(self.models.vhr.parameters(), lr=self.args.lr1)
        self.denoiser_optimizer_lidar = optimizer(self.models.lidar.parameters(), lr=self.args.lr1)
        self.denoiser_optimizer_hsi = optimizer(self.models.hsi.parameters(), lr=self.args.lr1)

    def init_classifier(self, loss=CosineSimilarityLoss(), optimizer=optim.Adam):
        self.models.init_classifier()
        self.classifier_criterion = loss
        self.classifier_optimizer = optimizer(self.models.classifier.parameters(), lr=self.args.lr2)

    def init_data_loader(self):
        dataset = Dataset_from_feature(args.dataset, args.if_small_dataset)
        self.data_loader_train, self.data_loader_test = SplitDataset(dataset, args.bs, 0.8)


def main(args):
    # setting init
    torch.set_float32_matmul_precision('medium')
    set_seed(args.seed)
    if args.trial_run == 0:
        """Data Report of small dataset"""
        data_report(args.dataset)
    elif args.trial_run == 1:
        # TODO: Encode the data
        dataset = None
        if args.dataset == 'SZU_R1':
            dataset = SZUTree_Dataset_R1_raw
        dataloader_raw = DataLoader(dataset, batch_size=args.bs, shuffle=True)
        loop = tqdm.tqdm(enumerate(dataloader_raw), total=len(dataloader_raw))
        pass
    elif args.trial_run == 2:
        # TODO: Encode + Diffusion + Classification
        pass
        # models = Models(args)
        # trainer = Trainer(models, args)
        # trainer.init_denoiser()
        # trainer.init_classifier()
        # trainer.init_data_loader()
        # trainer.to(CUDA0)
    else:
        print("\033[91m Trial Run设置错误！！！\033[0m")


if __name__ == '__main__':
    args = get_argument_parse()
    main(args)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import cosine_annealing_schedule


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义判别器的网络结构
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 前向传播
        x = x.float()
        output = self.model(x)
        return output.view(-1, 1)


class Classifier(nn.Module):
    """A simple convolutional neural network with residual connections."""
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.upsample_factor = 2
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.1),
            nn.ReLU(),
            self._make_residual_block(16, 16),
            self._make_residual_block(16, 16),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            self._make_residual_block(32, 32),
            self._make_residual_block(32, 32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.3),
            nn.ReLU(),
            self._make_residual_block(64, 64),
            self._make_residual_block(64, 64)
        )
        self.upsample = nn.Upsample(scale_factor=self.upsample_factor, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        identity = x
        x = self.model(x)
        x = x + identity
        x = self.upsample(x)  # [batch_size, 64, 64, 64]
        x = self.upsample(x)  # [batch_size, 64, 128, 128]

        x = x[:, :, 48:80, 48:80]

        x = self.final(x)  # [batch_size, 1, 32, 32]
        return x


class InvariantGenerator(pl.LightningModule):
    def __init__(self, encoder_rgb, encoder_hsi, encoder_lidar, adapter, discriminator, classifier, args):
        super().__init__()
        self.encoder_rgb = encoder_rgb
        self.encoder_hsi = encoder_hsi
        self.encoder_lidar = encoder_lidar
        self.adapter = adapter
        self.discriminator = discriminator
        self.classifier = classifier
        self.ce = None
        self.args = args
        self.save_hyperparameters()
        self.modality_label_rgb = torch.nn.functional.one_hot(torch.tensor(1, dtype=torch.long), num_classes=3)
        self.modality_label_hsi = torch.nn.functional.one_hot(torch.tensor(2, dtype=torch.long), num_classes=3)
        self.modality_label_lidar = torch.nn.functional.one_hot(torch.tensor(3, dtype=torch.long), num_classes=3)
        # definition of modal one-hot label

    def forward(self, data_dict):
        real_rgb, real_hsi, real_lidar = data_dict['rgb'], data_dict['hsi'], data_dict['lidar']
        fake_rgb, fake_hsi, fake_lidar = self.adapter(real_rgb), self.adapter(real_hsi), self.adapter(real_lidar)
        return fake_rgb, fake_hsi, fake_lidar

    def _generator_loss(self, fake_output_rgb, fake_output_hsi, fake_output_lidar):
        loss1 = (F.cross_entropy(fake_output_rgb, self.modality_label_hsi) +
                 F.cross_entropy(fake_output_rgb, self.modality_label_lidar)) / 2
        loss2 = (F.cross_entropy(fake_output_hsi, self.modality_label_rgb) +
                 F.cross_entropy(fake_output_hsi, self.modality_label_lidar)) / 2
        loss3 = (F.cross_entropy(fake_output_lidar, self.modality_label_rgb) +
                 F.cross_entropy(fake_output_lidar, self.modality_label_hsi)) / 2
        return (loss1 + loss2 + loss3) / 3

    def _discriminator_loss(self, real_output_rgb, fake_output_rgb,
                            real_output_hsi, fake_output_hsi,
                            real_output_lidar, fake_output_lidar):
        loss1 = (F.cross_entropy(real_output_rgb, self.modality_label_rgb) +
                 F.cross_entropy(fake_output_rgb, self.modality_label_rgb)*0) / 1
        loss2 = (F.cross_entropy(real_output_hsi, self.modality_label_hsi) +
                 F.cross_entropy(fake_output_hsi, self.modality_label_hsi)*0) / 1
        loss3 = (F.cross_entropy(real_output_lidar, self.modality_label_lidar) +
                 F.cross_entropy(fake_output_lidar, self.modality_label_lidar)*0) / 1
        return (loss1 + loss2 + loss3) / 3

    def _cls_loss(self, cls_pred, gt, class_weight=None, label_smoothing=0.1):
        if class_weight:
            self.ce = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)
        else:
            self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss = self.ce(cls_pred, gt)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        data_dict, gt = batch
        # training every modality with self() forward
        fake_rgb, fake_hsi, fake_lidar = self(data_dict)
        if optimizer_idx == 0:
            real_output_rgb, fake_output_rgb = self.discriminator(data_dict['rgb']), self.discriminator(fake_rgb)
            real_output_hsi, fake_output_hsi = self.discriminator(data_dict['hsi']), self.discriminator(fake_hsi)
            real_output_lidar, fake_output_lidar = self.discriminator(data_dict['lidar']), self.discriminator(fake_lidar)
            disc_loss = self._discriminator_loss(real_output_rgb, fake_output_rgb,
                                                 real_output_hsi, fake_output_hsi,
                                                 real_output_lidar, fake_output_lidar)
            self.log('disc_loss', disc_loss.item(), prog_bar=True, on_epoch=True, batch_size=self.args.bs)
            return disc_loss
        elif optimizer_idx == 1:
            fake_output_rgb = self.discriminator(fake_rgb)
            fake_output_hsi = self.discriminator(fake_hsi)
            fake_output_lidar = self.discriminator(fake_lidar)
            generate_loss = self._generator_loss(fake_output_rgb, fake_output_hsi, fake_output_lidar)
            self.log('gen_loss', generate_loss.item(), prog_bar=True, on_epoch=True, batch_size=self.args.bs)
            return generate_loss
        elif optimizer_idx == 2:
            cls_pred = self.classifier(fake_rgb, fake_hsi)
            cls_loss = self._cls_loss(cls_pred, gt)
            self.log('cls_loss', cls_loss, prog_bar=True, batch_size=self.args.bs)
            self.log('cls_acc',
                     (cls_pred.argmax(dim=1) == gt).float().mean(),
                     prog_bar=True)
            return cls_loss

    def configure_optimizers(self):
        lr_gan = self.args.lr3
        lr_cls = self.args.lr2
        betas = self.args.betas

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_gan, betas=betas)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_gan, betas=betas)
        opt_c = torch.optim.Adam(self.classifier.parameters(), lr=lr_cls)

        # 返回三个优化器，按顺序执行
        return [opt_d, opt_g, opt_c], []

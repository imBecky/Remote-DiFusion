import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models

import utils

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PreTrainedEncoder(nn.Module):
    def __init__(self):
        super(PreTrainedEncoder, self).__init__()
        self.encoder = self.gen_encoder()
        # self.reshape = Reshape((1, 32, 32))

    def gen_encoder(self):
        encoder = models.mobilenet_v2(pretrained=True).features
        encoder.eval()
        encoder = nn.Sequential(*list(encoder.children())[1:])
        # print(encoder[-1])
        # encoder[-1][0] = nn.Linear(320, 1024, bias=True)
        return encoder

    def forward(self, x):
        x = x.float()
        input_dim = x.shape[1]
        head = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True)
        ).to(CUDA0)
        x = head(x)
        x = self.encoder(x)
        # x = self.reshape(x)
        return x

# class MyClassifier(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MyClassifier, self).__init__()
#         self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)


class MyDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MyDecoder, self).__init__()
        self.decoder = nn.Sequential(
            # (1280, 16, 16)->(512, 16, 16)
            nn.Conv2d(1280, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            # (256, 16, 16) -> (128, 32, 32)
            nn.ConvTranspose2d(64, 32, (4,4), (2,2), (1,1)),
            nn.ReLU(),
            # (128, 32, 32)->(64, 64, 64)
            nn.ConvTranspose2d(32, 16, (4,4), (2,2), (1,1)),
            nn.ReLU(),
            # (64, 64, 64)->(32, 128, 128)
            nn.ConvTranspose2d(16, 8, (4,4), (2,2), (1,1)),
            nn.ReLU(),
            # (32, 128, 128)->(16, 256, 256)
            nn.ConvTranspose2d(8, 2, (4,4), (2,2), (1,1)),
            nn.ReLU(),
        )

    def forward(self, x):
        result = self.decoder(x)
        return result



class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),  # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU()
        )

        # 潜在空间的均值和方差
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.Sigmoid()  # 将输出限制在[0,1]范围内
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # 编码
        mu, log_var = self.encode(x)

        # 重参数化
        z = self.reparameterize(mu, log_var)

        # 解码
        x_recon = self.decode(z)

        return x_recon


class Discriminator(nn.Module):
    def __init__(self, cls_num=3):
        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
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
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 添加全局平均池化和分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(512, cls_num),
        )

    def forward(self, x):
        x = x.float()
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class Classifier(nn.Module):
    """A simple convolutional neural network with residual connections."""

    def __init__(self, pos_shape, latent_dim, num_heads=1):
        super(Classifier, self).__init__()
        self.pos_shape = pos_shape
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, bias=False)
        self.pos_encoding = nn.Parameter(torch.randn(1, latent_dim, pos_shape, pos_shape))

        # 修改后的上采样部分
        self.initial_upsample = nn.Sequential(
            # nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),  # 32x32 → 256x256
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)  # 增加通道数
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.1),
            nn.ReLU(),
            self._make_residual_block(32, 32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            self._make_residual_block(64, 64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.3),
            nn.ReLU(),
            self._make_residual_block(128, 128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 21, kernel_size=(1, 1))  # 输出单通道分类结果
        )
    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.1),
            nn.ReLU()
        )

    def forward(self, rgb, hsi, lidar):
        batch_size = rgb.shape[0]
        embed_dim = rgb.shape[1]

        # 添加位置编码
        rgb = rgb + self.pos_encoding
        hsi = hsi + self.pos_encoding
        lidar = lidar + self.pos_encoding

        # 展平空间维度 [B, embed_dim, 32 * 32]
        rgb = rgb.flatten(2).permute(2, 0, 1)
        hsi = hsi.flatten(2).permute(2, 0, 1)
        lidar = lidar.flatten(2).permute(2, 0, 1)

        # 拼接模态
        modal_cat = torch.cat([rgb, hsi, lidar], dim=0)  # [3*w*h, B, dim]

        # 注意力机制
        attn_output, _ = self.attention(modal_cat, modal_cat, modal_cat)

        # 取均值并恢复形状
        attn_output = attn_output.view(3, self.pos_shape * self.pos_shape, batch_size, embed_dim)
        attn_output = torch.mean(attn_output, dim=0)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, 1, 32, 32)

        # 初始上采样到256x256
        x = self.initial_upsample(attn_output)

        # classify
        x = self.classifier(x)
        x = x.squeeze()
        return x


class InvariantGenerator(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.adapter = VAE()
        self.discriminator = Discriminator()
        self.classifier = Classifier(32)
        self.ce = None
        self.args = args
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.modality_label_rgb = torch.zeros(args.bs, dtype=torch.long).to(CUDA0)
        self.modality_label_hsi = torch.ones(args.bs, dtype=torch.long).to(CUDA0)
        self.modality_label_lidar = (torch.ones(args.bs, dtype=torch.long) * 2).to(CUDA0)
        # definition of modal one-hot label
        self.encoder = PreTrainedEncoder()

    def forward(self, data_dict):
        real_rgb, real_hsi, real_lidar = data_dict['rgb'].float(), data_dict['hsi'].float(), data_dict['lidar'].float()
        real_rgb = self.encoder(real_rgb)
        real_hsi = self.encoder(real_hsi)
        real_lidar = self.encoder(real_lidar)
        fake_rgb, fake_hsi, fake_lidar = self.adapter(real_rgb), self.adapter(real_hsi), self.adapter(real_lidar)
        return real_rgb, real_hsi, real_lidar, fake_rgb, fake_hsi, fake_lidar

    def _generator_loss(self, fake_output_rgb, fake_output_hsi, fake_output_lidar):
        loss1 = (F.cross_entropy(fake_output_rgb, self.modality_label_rgb) +
                 F.cross_entropy(fake_output_rgb, self.modality_label_hsi) +
                 F.cross_entropy(fake_output_rgb, self.modality_label_lidar)) / 3
        loss2 = (F.cross_entropy(fake_output_hsi, self.modality_label_rgb) +
                 F.cross_entropy(fake_output_hsi, self.modality_label_hsi) +
                 F.cross_entropy(fake_output_hsi, self.modality_label_lidar)) / 3
        loss3 = (F.cross_entropy(fake_output_lidar, self.modality_label_rgb) +
                 F.cross_entropy(fake_output_lidar, self.modality_label_hsi) +
                 F.cross_entropy(fake_output_lidar, self.modality_label_lidar)) / 3
        return (loss1 + loss2 + loss3) / 3

    def _discriminator_loss(self, real_output_rgb, fake_output_rgb,
                            real_output_hsi, fake_output_hsi,
                            real_output_lidar, fake_output_lidar):
        loss1 = (F.cross_entropy(real_output_rgb, self.modality_label_rgb) +
                 F.cross_entropy(fake_output_rgb, self.modality_label_rgb) * 0) / 1
        loss2 = (F.cross_entropy(real_output_hsi, self.modality_label_hsi) +
                 F.cross_entropy(fake_output_hsi, self.modality_label_hsi) * 0) / 1
        loss3 = (F.cross_entropy(real_output_lidar, self.modality_label_lidar) +
                 F.cross_entropy(fake_output_lidar, self.modality_label_lidar) * 0) / 1
        return (loss1 + loss2 + loss3) / 3

    def _cls_loss(self, cls_pred, gt, class_weight=None, label_smoothing=0.1):
        if class_weight:
            self.ce = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)
        else:
            self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss = self.ce(cls_pred, gt)
        return loss

    def training_step(self, batch, batch_idx):
        data_dict, gt = batch
        gt = gt.long()
        opt_d, opt_g, opt_c = self.optimizers()
        # training every modality with self() forward
        encoded_rgb, encoded_hsi, encoded_lidar, fake_rgb, fake_hsi, fake_lidar = self(data_dict)
        # --- 训练判别器 (optimizer_idx=0) ---
        # 清零梯度
        opt_d.zero_grad()

        # 计算判别器损失
        real_output_rgb, fake_output_rgb = self.discriminator(encoded_rgb.detach()), \
            self.discriminator(fake_rgb.detach())
        real_output_hsi, fake_output_hsi = self.discriminator(encoded_hsi.detach()), \
            self.discriminator(fake_hsi.detach())
        real_output_lidar, fake_output_lidar = self.discriminator(encoded_lidar.detach()), \
            self.discriminator(fake_lidar.detach())
        disc_loss = self._discriminator_loss(real_output_rgb, fake_output_rgb,
                                             real_output_hsi, fake_output_hsi,
                                             real_output_lidar, fake_output_lidar)

        # 反向传播并更新参数
        self.manual_backward(disc_loss)
        opt_d.step()

        self.log('disc_loss', disc_loss.item(), prog_bar=True)

        # --- 训练生成器 (optimizer_idx=1) ---
        opt_g.zero_grad()

        fake_output_rgb = self.discriminator(fake_rgb)
        fake_output_hsi = self.discriminator(fake_hsi)
        fake_output_lidar = self.discriminator(fake_lidar)
        generate_loss = self._generator_loss(fake_output_rgb, fake_output_hsi, fake_output_lidar)

        self.manual_backward(generate_loss)
        opt_g.step()

        self.log('gen_loss', generate_loss.item(), prog_bar=True)

        # --- 训练分类器 (optimizer_idx=2) ---
        opt_c.zero_grad()
        _, _, _, fake_rgb, fake_hsi, fake_lidar = self(data_dict)  # Using new graph
        cls_pred = self.classifier(fake_rgb, fake_hsi, fake_lidar).squeeze(0)
        cls_loss = self._cls_loss(cls_pred, gt)

        self.manual_backward(cls_loss)
        opt_c.step()

        self.log('cls_loss', cls_loss, prog_bar=True)
        self.log('cls_acc', (cls_pred.argmax(dim=1) == gt).float().mean())

        return {
            'disc_loss': disc_loss,
            'gen_loss': generate_loss,
            'cls_loss': cls_loss
        }

    def validation_step(self, batch, batch_idx):
        data_dict, gt = batch
        gt = gt.long()

        # 前向传播
        _, _, _, fake_rgb, fake_hsi, fake_lidar = self(data_dict)

        # 计算分类结果
        cls_pred = self.classifier(fake_rgb, fake_hsi, fake_lidar).squeeze(0)
        cls_loss = self._cls_loss(cls_pred, gt)

        # 计算准确率
        acc = (cls_pred.argmax(dim=1) == gt).float().mean()

        # 记录指标
        self.log('val_cls_loss', cls_loss, prog_bar=True)
        self.log('val_cls_acc', acc, prog_bar=True)

        return {
            'val_cls_loss': cls_loss,
            'val_cls_acc': acc
        }

    def test_step(self, batch, batch_idx):
        data_dict, gt = batch
        gt = gt.long()

        # 前向传播
        _, _, _, fake_rgb, fake_hsi, fake_lidar = self(data_dict)

        # 计算分类结果
        cls_pred = self.classifier(fake_rgb, fake_hsi, fake_lidar).squeeze(0)
        cls_loss = self._cls_loss(cls_pred, gt)

        # 计算准确率
        acc = (cls_pred.argmax(dim=1) == gt).float().mean()

        # 记录指标
        self.log('test_cls_loss', cls_loss, prog_bar=True)
        self.log('test_cls_acc', acc, prog_bar=True)

        return {
            'test_cls_loss': cls_loss,
            'test_cls_acc': acc
        }

    def configure_optimizers(self):
        lr_gan = self.args.lr3
        lr_cls = self.args.lr2
        betas = self.args.betas

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_gan, betas=betas)
        opt_g = torch.optim.Adam(self.adapter.parameters(), lr=lr_gan, betas=betas)
        opt_c = torch.optim.Adam(self.classifier.parameters(), lr=lr_cls)

        # 返回三个优化器，按顺序执行
        return [opt_d, opt_g, opt_c]


class Module2(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PreTrainedEncoder()
        self.decoder_rgb = MyDecoder(1024)
        self.decoder_hsi = MyDecoder(1024)
        self.decoder_lidar = MyDecoder(1024)
        self.classifier = Classifier(256, 2)
        self.ce = None # cross entropy loss

    def forward(self, data_dict):
        rgb, hsi, lidar = data_dict['rgb'].float(), data_dict['hsi'].float(), data_dict['lidar'].float()
        encoded_rgb, encoded_hsi, encoded_lidar = self.encoder(rgb), self.encoder(hsi), self.encoder(lidar)
        decoded_rgb = self.decoder_rgb(encoded_rgb)
        decoded_hsi = self.decoder_hsi(encoded_hsi)
        decoded_lidar = self.decoder_lidar(encoded_lidar)
        pred = self.classifier(decoded_rgb, decoded_hsi, decoded_lidar)
        return pred

    def _cls_loss(self, cls_pred, gt, class_weight=None, label_smoothing=0.1):
        if class_weight:
            self.ce = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)
        else:
            self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss = self.ce(cls_pred, gt)
        return loss

    def training_step(self, batch, batch_idx):
        data_dict, gt = batch
        gt = gt.long()
        pred_rgb, pred_hsi, pred_lidar = self.forward(data_dict)
        loss_rgb = self._cls_loss(pred_rgb, gt)
        loss_hsi = self._cls_loss(pred_hsi, gt)
        loss_lidar = self._cls_loss(pred_lidar, gt)
        loss = (loss_rgb + loss_hsi + loss_lidar) / 3
        self.log('train_loss', loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data_dict, gt = batch
        gt = gt.long()
        pred_rgb, pred_hsi, pred_lidar = self.forward(data_dict)
        loss_rgb = self._cls_loss(pred_rgb, gt)
        loss_hsi = self._cls_loss(pred_hsi, gt)
        loss_lidar = self._cls_loss(pred_lidar, gt)
        loss = (loss_rgb + loss_hsi + loss_lidar) / 3
        acc_rgb = (pred_rgb.argmax(dim=1) == gt).float().mean()
        acc_hsi = (pred_hsi.argmax(dim=1) == gt).float().mean()
        acc_lidar = (pred_lidar.argmax(dim=1) == gt).float().mean()
        avg_acc = (acc_rgb + acc_hsi + acc_lidar) / 3
        self.log('val_acc', avg_acc.item(), prog_bar=True)
        self.log('val_loss', loss.item(), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        data_dict, gt = batch
        gt = gt.long()
        pred_rgb, pred_hsi, pred_lidar = self.forward(data_dict)
        loss_rgb = self._cls_loss(pred_rgb, gt)
        loss_hsi = self._cls_loss(pred_hsi, gt)
        loss_lidar = self._cls_loss(pred_lidar, gt)
        loss = (loss_rgb + loss_hsi + loss_lidar) / 3
        acc_rgb = (pred_rgb.argmax(dim=1) == gt).float().mean()
        acc_hsi = (pred_hsi.argmax(dim=1) == gt).float().mean()
        acc_lidar = (pred_lidar.argmax(dim=1) == gt).float().mean()
        avg_acc = (acc_rgb + acc_hsi + acc_lidar) / 3
        self.log('test_acc', avg_acc.item(), prog_bar=True)
        self.log('test_loss', loss.item(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
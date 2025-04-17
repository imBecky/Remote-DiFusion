import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import cosine_annealing_schedule


class InvariantGenerator(pl.LightningModule):
    def __init__(self, encoder, adapter, generator, discriminator, args):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.generator = generator
        self.discriminator = discriminator
        self.args = args
        # definition of modal one-hot label

    def forward(self, x, cls):
        feature = self.encoder(x)
        invariant_feature = self.adapter(feature)
        epsilon = torch.randn_like(feature)
        generated_feature, epsilon_hat = self.generator(invariant_feature, cls, epsilon)
        diff_loss = nn.MSELoss(epsilon, epsilon_hat)
        disc_output = self.discriminator(generated_feature)
        return generated_feature, disc_output, diff_loss

    def _discriminator_loss(self, output_rgb, output_hsi, output_lidar):
        modality_label_rgb = torch.nn.functional.one_hot(torch.tensor(1, dtype=torch.long), num_classes=3)
        modality_label_hsi = torch.nn.functional.one_hot(torch.tensor(2, dtype=torch.long), num_classes=3)
        modality_label_lidar = torch.nn.functional.one_hot(torch.tensor(3, dtype=torch.long), num_classes=3)
        loss_rgb = F.cross_entropy(output_rgb, modality_label_rgb)
        loss_hsi = F.cross_entropy(output_hsi, modality_label_hsi)
        loss_lidar = F.cross_entropy(output_lidar, modality_label_lidar)
        total_loss = (loss_rgb + loss_hsi + loss_lidar) / 3.0
        return total_loss

    def diffusion_parameters(self):
        T = self.args.T
        betas = cosine_annealing_schedule(T, 0.1)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_prod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_prod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)

    def training_step(self, batch, batch_idx):
        data_dict, label = batch
        generated_feature_rgb, disc_output_rgb, diff_loss_rgb = self(data_dict['rgb'], label)
        generated_feature_hsi, disc_output_hsi, diff_loss_hsi = self(data_dict['hsi'], label)
        generated_feature_lidar, disc_output_lidar, diff_loss_lidar = self(data_dict['lidar'], label)
        disc_loss = self._discriminator_loss(disc_output_rgb, disc_output_hsi, disc_output_lidar)
        generate_loss = (diff_loss_rgb + diff_loss_hsi + diff_loss_lidar) / 3.0



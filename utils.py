import torch
import random
import os
import importlib
import numpy as np
import cv2
import torch.nn as nn
from inspect import isfunction
import math
import warnings
from einops.layers.torch import Rearrange
from scipy.linalg import sqrtm
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from torch import Tensor
from typing import Optional, Tuple
from torch.nn import grad  # noqa: F401
from torch._torch_docs import sparse_support_notes


HSI_SHAPE = (50, 4172, 1202)   # (band, width, height)
new_shape = (50, 8344, 2404)
linear = torch.nn.Linear


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


DEVICE = try_gpu()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_hsi_narray(path):
    with open(path, 'r') as file:
        # [:223] is the description of the data
        data_string = file.read()[223:]
        data_array = np.array(list(map(int, data_string.split())), dtype=int)  # transfer str to np array
        data_array = data_array.reshape(HSI_SHAPE)
        data_array = data_array[:, 596:2980, 601:1202]
    data_array = torch.from_numpy(data_array).to(DEVICE)
    return data_array


def load_rgb_array(root, list):
    vhr = []
    for i in range(len(list)):
        img = np.array(cv2.imread(root + list[i], cv2.IMREAD_UNCHANGED))
        img = np.transpose(img, (2, 1, 0))
        vhr.append((img))
    vhr = np.concatenate(vhr, axis=1)
    vhr = torch.from_numpy(vhr).to(DEVICE)
    return vhr


def base_loader(path):
    with open(path, 'r') as file:
        data_string = file.read()
        base = np.array(list(map(float, data_string.split())))
        base = np.reshape(base, (8344, 2404))
        base = torch.from_numpy(base).to(DEVICE)
    return base


def load_lidar_raster(path):
    count = 0
    with open(path, 'r') as file:
        data_strings = file.readlines()
        raster = np.array(list(map(float, data_strings[0].split())))[:, np.newaxis]
        for line in data_strings[1:-1]:
            data_array = np.array(list(map(float, line.split())))
            data_array = data_array[:, np.newaxis]
            indices = np.where(data_array == 1000)
            if len(indices[0]) != 0:
                data_array[indices[0], 0] = -16
            raster = np.concatenate((raster, data_array), axis=1)
        # data_array = np.array(list(map(float, data_string.split())))
        raster = torch.from_numpy(raster).to(DEVICE)
        return raster


def ground_truth_loader(path):
    with open(path, 'r') as file:
        data_string = file.read()
        base = np.array(list(map(int, data_string.split())))
        base = np.reshape(base, (4768, 1202))
        return base


class Reshape(nn.Module):
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        # Ensure that the input tensor size matches the product of new_sthape
        batch_size = x.size(0)
        num_elements = 1
        for dim in self.new_shape:
            num_elements *= dim

        if x.shape[1] != num_elements:
            raise ValueError("Total number of elements must be the same after reshape")
        return x.view(batch_size, *self.new_shape)


class EMA:
    def __init__(self, decay):
        self.decay = decay

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)


def default(val, d):
    """
    该函数的目的是提供一个简单的机制来获取给定变量的默认值。
    如果 val 存在，则返回该值。如果不存在，则使用 d 函数提供的默认值，
    或者如果 d 不是一个函数，则返回 d。
    :param val:需要判断的变量
    :param d:提供默认值的变量或函数
    :return:
    """
    if val is not None:
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        """
        残差连接模块
        :param fn: 激活函数类型
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        残差连接前馈
        :param x: 输入数据
        :param args:
        :param kwargs:
        :return: f(x) + x
        """
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def exists(x):
    """
    判断数值是否为空
    :param x: 输入数据
    :return: 如果不为空则True 反之则返回False
    """
    return x is not None


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    """
    下采样模块的作用是将输入张量的分辨率降低，通常用于在深度学习模型中对特征图进行降采样。
    在这个实现中，下采样操作的方式是使用一个 $2 \times 2$ 的最大池化操作，
    将输入张量的宽和高都缩小一半，然后再使用上述的变换和卷积操作得到输出张量。
    由于这个实现使用了形状变换操作，因此没有使用传统的卷积或池化操作进行下采样，
    从而避免了在下采样过程中丢失信息的问题。
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        # 将输入张量的形状由 (batch_size, channel, height, width) 变换为 (batch_size, channel * 4, height / 2, width / 2)
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # 对变换后的张量进行一个 $1 \times 1$ 的卷积操作，将通道数从 dim * 4（即变换后的通道数）降到 dim（即指定的输出通道数），得到输出张量。
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


def Upsample(dim, dim_out=None):
    """
    这个上采样模块的作用是将输入张量的尺寸在宽和高上放大 2 倍
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),            # 先使用最近邻填充将数据在长宽上翻倍
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),    # 再使用卷积对翻倍后的数据提取局部相关关系填充
    )


def calculate_fid(act1, act2):
    act1_flattened = act1.reshape(32, -1)
    act2_flattened = act2.reshape(32, -1)
    mu1, sigma1 = act1_flattened.mean(axis=0), np.cov(act1_flattened, rowvar=False)
    mu2, sigma2 = act2_flattened.mean(axis=0), np.cov(act2_flattened, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_features(root):
    hsi_feature = torch.load(root+'/hsi.pth')
    ndsm_feature = torch.load(root+'/ndsm.pth')
    rgb_feature = torch.load(root+'/rgb.pth')
    return hsi_feature, ndsm_feature, rgb_feature


def extract(a, t, x_shape):
    batch_size = x_shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def cosine_annealing_schedule(length_T, initial_beta, final=0.001):
    beta_schedual = initial_beta * (final / initial_beta) ** (0.5 * np.cos(np.pi * np.arange(length_T) / length_T))
    beta_schedual = torch.from_numpy(beta_schedual)
    return beta_schedual


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, input1, input2):
        # 计算两个输入向量的点积
        input1_normalized = F.normalize(input1, p=2, dim=1)
        input2_normalized = F.normalize(input2, p=2, dim=1)
        cosine_similarity = torch.sum(input1_normalized * input2_normalized, dim=1)

        # 计算余弦相似度损失
        loss = 1 - cosine_similarity
        return loss.mean()  # 返回损失的平均值


def split_gt(gt, patch_size, stride):
    # 计算每个维度可以分割成多少个patch
    patches_x = (gt.shape[1] - patch_size[0]) // stride[0] + 1
    patches_y = (gt.shape[0] - patch_size[1]) // stride[1] + 1

    # 初始化一个列表来存储所有的patches
    patches = []

    # 循环遍历每个patch的位置
    for i in range(patches_y):
        for j in range(patches_x):
            x_start = j * stride[0]
            y_start = i * stride[1]

            patch = gt[y_start:y_start + patch_size[1], x_start:x_start + patch_size[0]]

            patches.append(patch)

    patches = np.array(patches)

    return patches


class Dataset_from_feature(data.Dataset):
    def __init__(self, root, if_small_dataset, stride=(2, 2)):
        super(Dataset_from_feature, self).__init__()
        self.root = root
        self.stride = stride
        self.hsi, self.ndsm, self.rgb, self.gt = self._get_data(if_small_dataset)

    def _get_data(self, if_small_dataset):
        if if_small_dataset == 1:
            if 'small' not in self.root:
                root = self.root+'/small'
            else:
                root = self.root
            gt = torch.load(root + '/gt.pth', weights_only=False)
            gt = torch.from_numpy(gt)
            gt = torch.unsqueeze(gt, 1)
            feature_hsi = torch.load(root + '/hsi.pth', weights_only=False)
            feature_ndsm = torch.load(root + '/ndsm.pth', weights_only=False)
            feature_rgb = torch.load(root + '/rgb.pth', weights_only=False)
            for item in [feature_hsi, feature_ndsm, feature_rgb, gt]:
                print(item.shape)
            return feature_hsi[:33], feature_ndsm[:33], feature_rgb[:33], gt[:33]
        else:
            root = self.root + '/feature'
            gt = torch.load(root + '/gt.pth', weights_only=False)
            feature_hsi = torch.load(root + '/hsi.pth', weights_only=False)
            feature_ndsm = torch.load(root + '/ndsm.pth', weights_only=False)
            feature_rgb = torch.load(root + '/rgb.pth', weights_only=False)
            return feature_hsi, feature_ndsm, feature_rgb, gt

    def __getitem__(self, item):
        return self.hsi[item], self.ndsm[item], self.rgb[item], self.gt[item]

    def __len__(self):
        return len(self.hsi)


def SplitDataset(dataset, batch_size, ratio):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


def CalculateParameters(model_dict):
    for i, model in enumerate(model_dict):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'模型总参数数量: {num_params}')

class MultiheadAttention(nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.parameter.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.parameter.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.parameter.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.parameter.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.parameter.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        # self.out_proj = torch.nn.Module.NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        if add_bias_kv:
            self.bias_k = nn.parameter.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.parameter.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
                or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif nn._is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(torch.nn.activation._check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(torch.nn.activation._arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            with torch.cuda.amp.autocast():
                attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=False,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal)
        else:
            with torch.cuda.amp.autocast():
                attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type


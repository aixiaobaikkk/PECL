from __future__ import division, print_function
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from einops.layers.torch import Rearrange
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


# 定义一个多层感知器神经网络类
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # Dropout层

    def forward(self, x):
        x = self.fc1(x)  # 输入经过第一个全连接层
        x = self.act(x)  # 激活函数
        x = self.drop(x)  # Dropout层
        x = self.fc2(x)  # 经过第二个全连接层
        x = self.drop(x)  # Dropout层
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 定义一个卷积多层感知器神经网络类
class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)  # 输入卷积层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)  # 输出卷积层
        self.drop = nn.Dropout(drop)  # Dropout层

    def forward(self, x):
        x = self.fc1(x)  # 输入经过第一个卷积层
        x = self.act(x)  # 激活函数
        x = self.drop(x)  # Dropout层
        x = self.fc2(x)  # 经过第二个卷积层
        x = self.drop(x)  # Dropout层
        return x


class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.upsample = nn.Upsample(scale_factor=sr_ratio, mode='nearest')
        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAgg(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # global layer_scale
        # self.ls = layer_scale

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)

        return x


class LG_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()

        if sr_ratio > 1:
            self.LocalAgg = LocalAgg(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                     act_layer, norm_layer)
        else:
            self.LocalAgg = nn.Identity()

        self.SelfAttn = SelfAttn(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                                 norm_layer, sr_ratio)

    def forward(self, x):
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)

class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out

class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out

class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W
class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # print(x.shape,"indual")
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)

        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)
        mx = add3 + mlp2

        return mx


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class LUCFEncoder(nn.Module):
    def __init__(self, params):
        super(LUCFEncoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.lg_en1 = LG_Block(dim=self.ft_chns[1], num_heads=3, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=self.dropout[1], sr_ratio=4, drop_path=0.1)
        self.lg_en2 = LG_Block(dim=self.ft_chns[2], num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=self.dropout[2], sr_ratio=2, drop_path=0.1)
        self.lg_en3 = LG_Block(dim=self.ft_chns[3], num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=self.dropout[3], sr_ratio=2, drop_path=0.1)
        self.lg_en4 = LG_Block(dim=self.ft_chns[4], num_heads=24, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=self.dropout[4], sr_ratio=1, drop_path=0.1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x1 = self.lg_en1(x1)
        x2 = self.down2(x1)
        x2 = self.lg_en2(x2)
        x3 = self.down3(x2)
        x3 = self.lg_en3(x3)
        x4 = self.down4(x3)
        x4 = self.lg_en4(x4)
        return [x0, x1, x2, x3, x4]


class CNNEncoder(nn.Module):
    def __init__(self, params):
        super(CNNEncoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]



class TransFormerEncoder(nn.Module):
    def __init__(self, image_size, in_chns, in_dim, key_dim, value_dim, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [3, 3, 3, 3]
        strides = [2, 2, 2, 2]
        padding_sizes = [1, 1, 1, 1]

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]

        self.in_conv = ConvBlock(in_chns, 16, 0.05)
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 16, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 2, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )


        self.patch_embed4 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[3], strides[3], padding_sizes[3], in_dim[2], in_dim[3]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [DualTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [DualTransformerBlock(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [DualTransformerBlock(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

        self.block4 = nn.ModuleList(
            [DualTransformerBlock(in_dim[3], key_dim[3], value_dim[3], head_count, token_mlp) for _ in range(layers[3])]
        )
        self.norm4 = nn.LayerNorm(in_dim[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        x = self.in_conv(x)
        outs.append(x)
        # stage 1
        x, H, W = self.patch_embed1(x)

        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class ShareDecoder(nn.Module):
    def __init__(self, params):
        super(ShareDecoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_head1 = nn.Conv2d(self.ft_chns[3],  self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.ft_chns[2],  self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.ft_chns[1],  self.n_class, 1)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)


    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        p = []
        p.append(self.out_head1(x))
        x = self.up2(x, x2)
        p.append( self.out_head2(x))
        x = self.up3(x, x1)
        p.append( self.out_head3(x))
        x = self.up4(x, x0)
        p.append(self.out_conv(x))

        return p

# class TriNet(nn.Module):
#     def __init__(self, in_chns, class_num):
#         super(TriNet, self).__init__()
#
#         params = {'in_chns': in_chns,
#                   'feature_chns': [16, 48, 96, 192, 384],
#                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
#                   'class_num': class_num,
#                   'bilinear': False,
#                   'acti_func': 'relu',
#                   'head_count': 1,
#                   'token_mlp_mode': "mix_skip",
#                   'layers': [2, 2, 2, 2]
#                   }
#
#         self.CNNEncoder = CNNEncoder(params)
#         self.LUCFEncoder = LUCFEncoder(params)
#         self.TransFormerEncoder =TransFormerEncoder(
#             image_size=224,
#             in_chns=in_chns,
#             in_dim=params['feature_chns'][1:],
#             key_dim=params['feature_chns'][1:],
#             value_dim=params['feature_chns'][1:],
#             layers=params['layers'],
#             head_count=params['head_count'],
#             token_mlp=params['token_mlp_mode'],
#         )
#
#
#         self.decoder = ShareDecoder(params)
#
#     def forward(self, x):
#         CNNFeature = self.CNNEncoder(x)
#         TransFormerFeature = self.TransFormerEncoder(x)
#         LUCFFeature = self.LUCFEncoder(x)
#
#         output = self.decoder(feature)
#         output[0] = F.interpolate(output[0], scale_factor=8, mode='bilinear')
#         output[1] = F.interpolate(output[1], scale_factor=4, mode='bilinear')
#         output[2] = F.interpolate(output[2], scale_factor=2, mode='bilinear')
#         output[3] = F.interpolate(output[3], scale_factor=1, mode='bilinear')
#
#         return output[0] , output[1] , output[2] ,output[3]

class TriNetCNN(nn.Module):
    def __init__(self, in_chns, class_num):
        super(TriNetCNN, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 48, 96, 192, 384],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'head_count': 1,
                  'token_mlp_mode': "mix_skip",
                  'layers': [2, 2, 2, 2]
                  }

        self.CNNEncoder = CNNEncoder(params)
        self.decoder = ShareDecoder(params)

    def forward(self, x):
        CNNFeature = self.CNNEncoder(x)
        output = self.decoder(CNNFeature)
        output[0] = F.interpolate(output[0], scale_factor=8, mode='bilinear')
        output[1] = F.interpolate(output[1], scale_factor=4, mode='bilinear')
        output[2] = F.interpolate(output[2], scale_factor=2, mode='bilinear')
        output[3] = F.interpolate(output[3], scale_factor=1, mode='bilinear')

        return output[0] , output[1] , output[2] ,output[3]

class TriNetLUCFormer(nn.Module):
    def __init__(self, in_chns=3, class_num=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 48, 96, 192, 384],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        # Encoder

        dims, key_dim, value_dim, layers = [[48, 96, 192, 384], [48, 96, 192, 384], [48, 96, 192, 384], [2, 2, 2, 2]]
        self.backbone = TransFormerEncoder(
            image_size=224,
            in_chns=in_chns,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
        )

        self.decoder = ShareDecoder(params)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feature = self.backbone(x)
        output = self.decoder(feature)

        output[0] = F.interpolate(output[0], scale_factor=8, mode='bilinear')
        output[1] = F.interpolate(output[1], scale_factor=4, mode='bilinear')
        output[2] = F.interpolate(output[2], scale_factor=2, mode='bilinear')
        output[3] = F.interpolate(output[3], scale_factor=1, mode='bilinear')

        return output[0], output[1], output[2], output[3]


class TriNetLUCFNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(TriNetLUCFNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 48, 96, 192, 384],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'head_count': 1,
                  'token_mlp_mode': "mix_skip",
                  'layers': [2, 2, 2, 2]
                  }
        self.LUCFEncoder = LUCFEncoder(params)
        self.decoder = ShareDecoder(params)

    def forward(self, x):
        LUCFFeature = self.LUCFEncoder(x)
        output = self.decoder(LUCFFeature)
        output[0] = F.interpolate(output[0], scale_factor=8, mode='bilinear')
        output[1] = F.interpolate(output[1], scale_factor=4, mode='bilinear')
        output[2] = F.interpolate(output[2], scale_factor=2, mode='bilinear')
        output[3] = F.interpolate(output[3], scale_factor=1, mode='bilinear')
        return output[0], output[1], output[2], output[3]



class TriNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(TriNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 48, 96, 192, 384],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'head_count': 1,
                  'token_mlp_mode': "mix_skip",
                  'layers': [2, 2, 2, 2]
                  }
        self.LUCFEncoder = LUCFEncoder(params)
        self.CNNEncoder = CNNEncoder(params)


        dims, key_dim, value_dim, layers = [[48, 96, 192, 384], [48, 96, 192, 384], [48, 96, 192, 384], [2, 2, 2, 2]]
        self.TransFormerEncoder = TransFormerEncoder(
            image_size=224,
            in_chns=3,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=1,
            token_mlp="mix_skip",
        )

        self.decoder = ShareDecoder(params)

    def forward(self, x):

        feature3 = self.LUCFEncoder(x)
        feature2 = self.TransFormerEncoder(x.repeat(1, 3, 1, 1))
        feature1 = self.CNNEncoder(x)
        output1 = self.decoder(feature1)
        output2 = self.decoder(feature2)
        output3 = self.decoder(feature3)

        return output1, output2, output3
        #
        # output[0] = F.interpolate(output[0], scale_factor=8, mode='bilinear')
        # output[1] = F.interpolate(output[1], scale_factor=4, mode='bilinear')
        # output[2] = F.interpolate(output[2], scale_factor=2, mode='bilinear')
        # output[3] = F.interpolate(output[3], scale_factor=1, mode='bilinear')
        #
        #
        # return output[0], output[1], output[2], output[3]


if __name__ =="__main__":
    import torch
    from thop import profile
    # from pytorch_model_summary import summary
    m1 = TriNetCNN(in_chns=1, class_num=4)
    m2 = TriNetLUCFormer(in_chns=3, class_num=4)
    m3 = TriNetLUCFNet(in_chns=1, class_num=4)
    m_ = TriNet(in_chns=1, class_num=4)

    for m in [m_]:
        x = torch.rand(1,1,224,224)
        p1, p2, p3, p4 = m(x)

        model = m
        # print(summary(model, x, show_input=False, show_hierarchical=False))
        flops, params = profile(model, (x,))
        print('GFLOPs: ', flops/1000000000, 'params: ', params/1000000)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total parameters count", pytorch_total_params)


